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
#ifndef DETECTION_ENGINE_
#define DETECTION_ENGINE_

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
#include "inference_helper.h"
#include "bounding_box.h"


class DetectionEngine {
public:
    enum {
        kRetOk = 0,
        kRetErr = -1,
    };

    typedef struct Result_ {
        std::vector<BoundingBox> bbox_list;
        struct crop_ {
            int32_t x;
            int32_t y;
            int32_t w;
            int32_t h;
            crop_() : x(0), y(0), w(0), h(0) {}
        } crop;
        double                   time_pre_process;		// [msec]
        double                   time_inference;		// [msec]
        double                   time_post_process;	    // [msec]
        Result_() : time_pre_process(0), time_inference(0), time_post_process(0)
        {}
    } Result;

public:
    DetectionEngine(float threshold_box_confidence = 0.4f, float threshold_class_confidence = 0.2f, float threshold_nms_iou = 0.5f) {
        threshold_box_confidence_ = threshold_box_confidence;
        threshold_class_confidence_ = threshold_class_confidence;
        threshold_nms_iou_ = threshold_nms_iou;
    }
    ~DetectionEngine() {}
    int32_t Initialize(const std::string& work_dir, const int32_t num_threads);
    int32_t Finalize(void);
    int32_t Process(const cv::Mat& original_mat, Result& result);

private:
    int32_t ReadLabel(const std::string& filename, std::vector<std::string>& label_list);
    void GetBoundingBox(const float* data, float scale_x, float  scale_y, int32_t grid_w, int32_t grid_h, std::vector<BoundingBox>& bbox_list);

private:
    std::unique_ptr<InferenceHelper> inference_helper_;
    std::vector<InputTensorInfo> input_tensor_info_list_;
    std::vector<OutputTensorInfo> output_tensor_info_list_;
    std::vector<std::string> label_list_;

    float threshold_box_confidence_;
    float threshold_class_confidence_;
    float threshold_nms_iou_;
};

#endif
