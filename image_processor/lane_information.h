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
#ifndef LANE_INFORMATION_H_
#define LANE_INFORMATION_H_

/* for general */
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <memory>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "lane_engine.h"

class LaneInformation {
public:
    enum {
        kRetOk = 0,
        kRetErr = -1,
    };

private:
    typedef struct LineCoeff_ {
        /* y = a * x^2 + b * x + c */
        /*   y = depth, x = horizontal on top view image */
        double a;
        double b;
        double c;
        LineCoeff_() : a(0), b(0), c(0) {}
        LineCoeff_(double _a, double _b, double _c) : a(_a), b(_b), c(_c) {}
    } LineCoeff;

public:
    LaneInformation(): time_pre_process(0), time_inference(0), time_post_process(0) {}
    ~LaneInformation() {}
    int32_t Initialize(const std::string& work_dir, const int32_t num_threads);
    int32_t Finalize(void);
    int32_t Process(const cv::Mat& mat, const cv::Mat& mat_transform);
    void Draw(cv::Mat& mat, cv::Mat& mat_topview);
    double GetTimePreProcess() { return time_pre_process; };
    double GetTimeInference() { return time_inference; };
    double GetTimePostProcess() { return time_post_process; };

private:
    cv::Scalar GetColorForLine(int32_t id);

private:
    LaneEngine lane_engine_;

    std::vector<std::vector<cv::Point2f>> normal_line_list_;
    std::vector<std::vector<cv::Point2f>> topview_line_list_;
    std::vector<LineCoeff> line_coeff_list_;
    std::vector<bool> line_valid_list_;
    std::vector<int32_t> line_det_cnt_list_;
    std::vector<int32_t> y_draw_start_list_;

    double time_pre_process;    // [msec]
    double time_inference;      // [msec]
    double time_post_process;   // [msec]

};

#endif
