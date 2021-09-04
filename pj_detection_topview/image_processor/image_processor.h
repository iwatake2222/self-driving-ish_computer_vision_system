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
#ifndef IMAGE_PROCESSOR_H_
#define IMAGE_PROCESSOR_H_

/* for general */
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "image_processor_if.h"
#include "detection_engine.h"
#include "tracker.h"
#include "camera_model.h"

namespace cv {
    class Mat;
};

class ImageProcessor : public ImageProcessorIf {
public:
    ImageProcessor();
    ~ImageProcessor() override;
    int32_t Initialize(const InputParam& input_param) override;
    int32_t Process(const cv::Mat& mat_original, Result& result) override;
    int32_t Finalize(void) override;
    int32_t Command(int32_t cmd) override;

private:
    void DrawFps(cv::Mat& mat, double time_inference, cv::Point pos, double font_scale, int32_t thickness, cv::Scalar color_front, cv::Scalar color_back, bool is_text_on_rect = true);
    cv::Scalar GetColorForId(int32_t id);
    int32_t ProcessObjectDetection(const cv::Mat& mat_original, DetectionEngine::Result& det_result);
    void DrawObjectDetection(cv::Mat& mat, cv::Mat& mat_topview, const DetectionEngine::Result& det_result);
    void ResetCamera(int32_t width, int32_t height, float fov_deg);
    void CreateTransformMat();
    void CreateTopViewMat(const cv::Mat& mat_original, cv::Mat& mat_topview);
    void GetCameraParameter(std::array<float, 3>& real_rvec, std::array<float, 3>& real_tvec, std::array<float, 3>& top_rvec, std::array<float, 3>& top_tvec);
    void SetCameraParameter(const std::array<float, 3>& real_rvec, const std::array<float, 3>& real_tvec, const std::array<float, 3>& top_rvec, const std::array<float, 3>& top_tvec);

private:
    int32_t frame_cnt;
    DetectionEngine m_detection_engine;
    Tracker m_tracker;
    CameraModel camera_real;
    CameraModel camera_top;
    cv::Mat mat_transform_;
};

#endif
