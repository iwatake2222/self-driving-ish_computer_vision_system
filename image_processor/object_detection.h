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
#ifndef OBJECT_DETECTION_H_
#define OBJECT_DETECTION_H_

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
#include "detection_engine.h"
#include "tracker.h"
#include "camera_model.h"

class ObjectDetection {
public:
    enum {
        kRetOk = 0,
        kRetErr = -1,
    };

public:
    ObjectDetection(): time_pre_process_(0), time_inference_(0), time_post_process_(0) {}
    ~ObjectDetection() {}
    int32_t Initialize(const std::string& work_dir, const int32_t num_threads);
    int32_t Finalize(void);
    int32_t Process(const cv::Mat& mat, const cv::Mat& mat_transform, CameraModel& camera);
    void Draw(cv::Mat& mat, cv::Mat& mat_topview);
    double GetTimePreProcess() { return time_pre_process_; };
    double GetTimeInference() { return time_inference_; };
    double GetTimePostProcess() { return time_post_process_; };

private:
    cv::Scalar GetColorForId(int32_t id);

private:
    DetectionEngine detection_engine_;
    Tracker tracker_;
    cv::Rect roi_;

    double time_pre_process_;    // [msec]
    double time_inference_;      // [msec]
    double time_post_process_;   // [msec]
};

#endif
