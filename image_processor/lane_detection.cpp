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

int32_t LaneDetection::Process(const cv::Mat& mat, const cv::Mat& mat_transform, CameraModel& camera)
{
    /* Run inference to get line (points) */
    LaneEngine::Result lane_result;
    if (lane_engine_.Process(mat, lane_result) != LaneEngine::kRetOk) {
        return kRetErr;
    }
    time_pre_process_ = lane_result.time_pre_process;
    time_inference_ = lane_result.time_inference;
    time_post_process_ = lane_result.time_post_process;

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

    /* Convert to ground plane */
    ground_line_list_.clear();
    for (const auto& line : normal_line_list_) {
        std::vector<cv::Point2f> ground_line;
        std::vector<cv::Point3f> ground_line_xyz;
        camera.ConvertImage2GroundPlane(line, ground_line_xyz);
        for (const auto& p : ground_line_xyz) {
            ground_line.push_back({ p.z, p.x });
        }
        ground_line_list_.push_back(ground_line);
    }

    /* Curve Fitting (y = ax^2 + bx + c, where y = depth, x = horizontal) */
    current_line_valid_list_.clear();
    std::vector<LineCoeff> current_line_coeff_list;
    for (auto line : ground_line_list_) {
        double a = 0, b = 0, c = 0;
        double error = 999;
        if (line.size() > 4 && std::abs(line[0].x - line[line.size() - 1].x) > 5) {
            /* At first, try to use quadratic regression if I have enough points and the points have enough length (more than 5m) */
            (void)CurveFitting::SolveQuadraticRegression(line, a, b, c);
            error = CurveFitting::ErrorMaxQuadraticRegression(line, a, b, c);
        }
        if (error > 0.1 && line.size() > 2) {
            /* Use linear regression, if I didn't use quadratic regression or the result of quadratic regression is not good (the maximum error > 0.1m) */
            (void)CurveFitting::SolveLinearRegression(line, b, c);
            error = CurveFitting::ErrorMaxLinearRegression(line, b, c);
            if (error > 0.1) {
                /* Regression failes is error is huge */
                a = 0;
                b = 0;
                c = 0;
            }
        }
        current_line_coeff_list.push_back({ a, b, c });
        if (a == 0 && b == 0 && c == 0) {
            current_line_valid_list_.push_back(false);
        } else {
            current_line_valid_list_.push_back(true);
        }
    }

    if (line_coeff_list_.empty()) {
        line_coeff_list_ = current_line_coeff_list;
        line_valid_list_ = current_line_valid_list_;
        line_det_cnt_list_.resize(topview_line_list_.size());
    }

    /* Update coeff with smoothing */
    for (int32_t line_index = 0; line_index < static_cast<int32_t>(line_coeff_list_.size()); line_index++) {
        if (current_line_valid_list_[line_index]) {
            float kMixRatio = 0.05f;
            if (!line_valid_list_[line_index]) {
                kMixRatio = 1.0f;   /* detect the line for the first time */
            } else if (line_det_cnt_list_[line_index] < 10) {
                kMixRatio = 0.2f;   /* the first few frames after the line is detected for the first time */
            }
            auto& line_coeff = line_coeff_list_[line_index];
            line_coeff.a = current_line_coeff_list[line_index].a * kMixRatio + line_coeff.a * (1.0 - kMixRatio);
            line_coeff.b = current_line_coeff_list[line_index].b * kMixRatio + line_coeff.b * (1.0 - kMixRatio);
            line_coeff.c = current_line_coeff_list[line_index].c * kMixRatio + line_coeff.c * (1.0 - kMixRatio);
        }
    }

    /* Check if line is (possibly) valid */
    for (int32_t line_index = 0; line_index < static_cast<int32_t>(current_line_valid_list_.size()); line_index++) {
        if (current_line_valid_list_[line_index]) {
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

    return kRetOk;
}

void LaneDetection::Draw(cv::Mat& mat, cv::Mat& mat_topview, CameraModel& camera)
{
    /*** Draw on NormalView ***/
    /* draw points */
    for (int32_t line_index = 0; line_index < static_cast<int32_t>(normal_line_list_.size()); line_index++) {
        const auto& line = normal_line_list_[line_index];
        for (const auto& p : line) {
            cv::circle(mat, p, 5, GetColorForLine(line_index), 2);
        }
    }

    /*** Draw on TopView ***/
    /* draw points */
    for (int32_t line_index = 0; line_index < static_cast<int32_t>(topview_line_list_.size()); line_index++) {
        const auto& line = topview_line_list_[line_index];
        for (const auto& p : line) {
            cv::circle(mat_topview, p, 5, GetColorForLine(line_index), 2);
        }
    }

    /* draw line */
    static constexpr float kLineIntervalMeter = 1.0f;
    static constexpr float kLineFarthestPointMeter[4] = { 10.0f, 15.0f, 15.0f, 10.0f };
    for (int32_t line_index = 0; line_index < static_cast<int32_t>(line_coeff_list_.size()); line_index++) {
        const auto& coeff = line_coeff_list_[line_index];
        if (line_valid_list_[line_index]) {
            for (float z = 0; z < kLineFarthestPointMeter[line_index]; z += kLineIntervalMeter) {
                float z0 = z;
                float z1 = current_line_valid_list_[line_index] ? z + kLineIntervalMeter : z + kLineIntervalMeter / 2;
                float x0 = static_cast<float>(coeff.a * z0 * z0 + coeff.b * z0 + coeff.c);
                float x1 = static_cast<float>(coeff.a * z1 * z1 + coeff.b * z1 + coeff.c);
                cv::Point2f image_point_0;
                cv::Point2f image_point_1;
                camera.ConvertWorld2Image({ x0, 0, z0 }, image_point_0);
                camera.ConvertWorld2Image({ x1, 0, z1 }, image_point_1);
                cv::line(mat_topview, image_point_0, image_point_1, GetColorForLine(line_index), 2);
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
