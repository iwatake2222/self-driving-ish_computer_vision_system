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
#include "camera_model.h"
#include "detection_engine.h"
#include "tracker.h"
#include "lane_engine.h"
#include "semantic_segmentation_engine.h"
#include "depth_engine.h"

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

    void ResetCamera(int32_t width = 0, int32_t height = 0, float fov_deg = 0) override;
    void GetCameraParameter(float& focal_length, std::array<float, 3>& real_rvec, std::array<float, 3>& real_tvec, std::array<float, 3>& top_rvec, std::array<float, 3>& top_tvec) override;
    void SetCameraParameter(float focal_length, const std::array<float, 3>& real_rvec, const std::array<float, 3>& real_tvec, const std::array<float, 3>& top_rvec, const std::array<float, 3>& top_tvec) override;

private:
    void DrawFps(cv::Mat& mat, double time_inference, cv::Point pos, double font_scale, int32_t thickness, cv::Scalar color_front, cv::Scalar color_back, bool is_text_on_rect = true);
    cv::Scalar GetColorForId(int32_t id);
    cv::Scalar GetColorForLine(int32_t id);
    cv::Scalar GetColorForSegmentation(int32_t id);
    void CreateTransformMat();
    void CreateTopViewMat(const cv::Mat& mat_original, cv::Mat& mat_topview);

    void DrawObjectDetection(cv::Mat& mat, cv::Mat& mat_topview, const DetectionEngine::Result& det_result);
    void DrawLaneDetection(cv::Mat& mat, cv::Mat& mat_topview, const LaneEngine::Result& lane_result);
    void DrawSegmentation(cv::Mat& mat_segmentation, const SemanticSegmentationEngine::Result& segmentation_result);
    void DrawDepth(cv::Mat& mat, const DepthEngine::Result& depth_result);


private:
    int32_t frame_cnt_;
    CameraModel camera_real_;
    CameraModel camera_top_;
    cv::Mat mat_transform_;

    DetectionEngine detection_engine_;
    Tracker tracker_;
    LaneEngine lane_engine_;
    SemanticSegmentationEngine segmentation_engine_;
    DepthEngine depth_engine_;

};

#endif
