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
#include "lane_detection.h"


/*** Macro ***/
#define TAG "LaneDetection"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)


/*** Function ***/
int32_t LaneDetection::Initialize(const std::string& work_dir, const int32_t num_threads)
{
    if (lane_engine_.Initialize(work_dir, num_threads) != LaneEngine::kRetOk) {
        lane_engine_.Finalize();
        return kRetErr;
    }
    return kRetOk;
}

int32_t LaneDetection::Finalize()
{
    if (lane_engine_.Finalize() != LaneEngine::kRetOk) {
        return kRetErr;
    }
    return kRetOk;
}

int32_t LaneDetection::Process(const cv::Mat& mat, const cv::Mat& mat_transform)
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
    std::vector<LineCoeff> current_line_coeff_list;
    std::vector<bool> current_line_valid_list;
    for (auto line : topview_line_list_) {
        for (auto& p : line) std::swap(p.x, p.y);
        double a = 0, b = 0, c = 0;
        if (line.size() > 2) {
            /* At first, try to use linear regression. In case it's not enough(error is big), use quadratic regression */
            (void)CurveFitting::SolveLinearRegression(line, b, c);
            if (CurveFitting::ErrorMaxLinearRegression(line, b, c) > 5 && line.size() > 4) {
                (void)CurveFitting::SolveQuadraticRegression(line, a, b, c);
            }
        }
        current_line_coeff_list.push_back({ a, b, c });
        if (a == 0 && b == 0 && c == 0) {
            current_line_valid_list.push_back(false);
        } else {
            current_line_valid_list.push_back(true);
        }
    }

    if (line_coeff_list_.empty()) {
        line_coeff_list_ = current_line_coeff_list;
        line_valid_list_ = current_line_valid_list;
        line_det_cnt_list_.resize(topview_line_list_.size());
        y_draw_start_list_.resize(topview_line_list_.size(), 9999);
    }

    /* Update coeff with smoothing */
    for (int32_t line_index = 0; line_index < line_coeff_list_.size(); line_index++) {
        if (current_line_valid_list[line_index]) {
            float kMixRatio = 0.05f;
            if (!line_valid_list_[line_index]) {
                kMixRatio = 1.0f;   /* detect the line at the first time */
            } else if (line_det_cnt_list_[line_index] < 10) {
                kMixRatio = 0.2f;   /* the first few frames after the line is detected at the first time */
            }
            auto& line_coeff = line_coeff_list_[line_index];
            line_coeff.a = current_line_coeff_list[line_index].a * kMixRatio + line_coeff.a * (1.0 - kMixRatio);
            line_coeff.b = current_line_coeff_list[line_index].b * kMixRatio + line_coeff.b * (1.0 - kMixRatio);
            line_coeff.c = current_line_coeff_list[line_index].c * kMixRatio + line_coeff.c * (1.0 - kMixRatio);
        }
    }

    /* Check if line is (possibly) valid */
    for (int32_t line_index = 0; line_index < current_line_valid_list.size(); line_index++) {
        if (current_line_valid_list[line_index]) {
            if (line_det_cnt_list_[line_index] < 0) {
                line_det_cnt_list_[line_index] = 0;
            } else {
                line_det_cnt_list_[line_index]++;
            }
            line_valid_list_[line_index] = true;
        } else {
            if (line_det_cnt_list_[line_index] > 0) {
                line_det_cnt_list_[line_index] = 0;
            } else {
                line_det_cnt_list_[line_index]--;
            }
            if (line_det_cnt_list_[line_index] < -40) {
                line_valid_list_[line_index] = false;
            }
        }
    }

    /* Store line start position */
    for (int32_t line_index = 0; line_index < topview_line_list_.size(); line_index++) {
        const auto& line = topview_line_list_[line_index];
        if (current_line_valid_list[line_index]) {
            float y_top = line[0].y;
            float y_bottom = line[line.size() - 1].y;
            float length = std::abs(y_bottom - y_top);
            y_draw_start_list_[line_index] = (std::max)(static_cast<int32_t>(y_top - length * 0.5), 0);
        }
    }

    return kRetOk;
}

void LaneDetection::Draw(cv::Mat& mat, cv::Mat& mat_topview)
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
        if (line_valid_list_[line_index]) {
            for (int32_t y = y_draw_start_list_[line_index]; y < mat.rows - kLineIntervalPx; y += kLineIntervalPx) {
                int32_t y0 = y;
                int32_t y1 = y + kLineIntervalPx;
                int32_t x0 = static_cast<int32_t>(coeff.a * y0 * y0 + coeff.b * y0 + coeff.c);
                int32_t x1 = static_cast<int32_t>(coeff.a * y1 * y1 + coeff.b * y1 + coeff.c);
                cv::line(mat_topview, cv::Point(x0, y0), cv::Point(x1, y1), GetColorForLine(line_index), 2);
            }
        }
    }
}

cv::Scalar LaneDetection::GetColorForLine(int32_t id)
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
