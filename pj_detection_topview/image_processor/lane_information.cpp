/* Copyright 2021 iwatake2222

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/*** Include ***/
/* for general */
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <numeric>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "common_helper.h"
#include "common_helper_cv.h"
#include "curve_fitting.h"
#include "lane_information.h"


/*** Macro ***/
#define TAG "LaneInformation"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)


/*** Function ***/
int32_t LaneInformation::Initialize(const std::string& work_dir, const int32_t num_threads)
{
    if (lane_engine_.Initialize(work_dir, num_threads) != LaneEngine::kRetOk) {
        lane_engine_.Finalize();
        return kRetErr;
    }
    return kRetOk;
}

int32_t LaneInformation::Finalize()
{
    if (lane_engine_.Finalize() != LaneEngine::kRetOk) {
        return kRetErr;
    }
    return kRetOk;
}

int32_t LaneInformation::Process(const cv::Mat& mat, const cv::Mat& mat_transform)
{
    /* Run inference to get line (points) */
    LaneEngine::Result lane_result;
    if (lane_engine_.Process(mat, lane_result) != LaneEngine::kRetOk) {
        return kRetErr;
    }
    time_pre_process = lane_result.time_pre_process;
    time_inference = lane_result.time_inference;
    time_post_process = lane_result.time_post_process;

    /* Save result as points */
    normal_line_list_.clear();
    for (const auto& line : lane_result.line_list) {
        std::vector<cv::Point2f> normal_line;
        for (const auto& p : line) {
            normal_line.push_back({ static_cast<float>(p.first), static_cast<float>(p.second) });
        }
        normal_line_list_.push_back(normal_line);
    }

    /* Convert to topview */
    topview_line_list_.clear();
    for (const auto& line : normal_line_list_) {
        std::vector<cv::Point2f> topview_line;
        if (line.size() > 0) {
            cv::perspectiveTransform(line, topview_line, mat_transform);
        }
        topview_line_list_.push_back(topview_line);
    }

    /* Curve Fitting (y = ax^2 + bx + c, where y = depth, x = horizontal) */
    std::vector<LineCoeff> curre_tline_coeff_list;
    for (auto line : topview_line_list_) {
        for (auto& p : line) std::swap(p.x, p.y);
        double a = 0, b = 0, c = 0;
        if (line.size() > 5) {
            (void)CurveFitting::SolveQuadraticRegression(line, a, b, c);
        }
        if (a == 0 && b == 0) {
            if (line.size() > 2) {
                (void)CurveFitting::SolveLinearRegression(line, b, c);
            }
        }
        curre_tline_coeff_list.push_back({ a, b, c });
    }

    line_coeff_list_ = curre_tline_coeff_list;

    return kRetOk;
}

void LaneInformation::Draw(cv::Mat& mat, cv::Mat& mat_topview)
{
    /*** Draw on NormalView ***/
    /* draw points */
    for (int32_t line_index = 0; line_index < normal_line_list_.size(); line_index++) {
        const auto& line = normal_line_list_[line_index];
        for (const auto& p : line) {
            cv::circle(mat, p, 5, GetColorForLine(line_index), 2);
        }
    }

    /*** Draw on TopView ***/
    /* draw points */
    for (int32_t line_index = 0; line_index < topview_line_list_.size(); line_index++) {
        const auto& line = topview_line_list_[line_index];
        for (const auto& p : line) {
            cv::circle(mat_topview, p, 5, GetColorForLine(line_index), 2);
        }
    }

    /* draw line */
    static constexpr int32_t kLineIntervalPx = 5;
    for (int32_t line_index = 0; line_index < topview_line_list_.size(); line_index++) {
        const auto& line = topview_line_list_[line_index];
        const auto& coeff = line_coeff_list_[line_index];
        if (line.size() >= 2) {
            int32_t y_start = static_cast<int32_t>(line[0].y - std::abs(line[0].y - line[1].y));
            for (int32_t y = y_start; y < mat.rows - kLineIntervalPx; y += kLineIntervalPx) {
                int32_t y0 = y;
                int32_t y1 = y + kLineIntervalPx;
                int32_t x0 = static_cast<int32_t>(coeff.a * y0 * y0 + coeff.b * y0 + coeff.c);
                int32_t x1 = static_cast<int32_t>(coeff.a * y1 * y1 + coeff.b * y1 + coeff.c);
                cv::line(mat_topview, cv::Point(x0, y0), cv::Point(x1, y1), GetColorForLine(line_index), 2);
            }
        }
    }
}

cv::Scalar LaneInformation::GetColorForLine(int32_t id)
{
    switch (id) {
    default:
    case 0:
        return CommonHelper::CreateCvColor(255, 255, 0);
    case 1:
        return CommonHelper::CreateCvColor(0, 255, 255);
    case 2:
        return CommonHelper::CreateCvColor(0, 255, 255);
    case 3:
        return CommonHelper::CreateCvColor(255, 255, 0);
    }
}
