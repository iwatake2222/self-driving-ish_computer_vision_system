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
#include <memory>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "common_helper.h"
#include "common_helper_cv.h"
#include "camera_model.h"
#include "bounding_box.h"
#include "detection_engine.h"
#include "tracker.h"
#include "lane_detection.h"
#include "depth_engine.h"

#include "image_processor_if.h"
#include "image_processor.h"


/*** Macro ***/
#define TAG "ImageProcessor"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

#define COLOR_BG  CommonHelper::CreateCvColor(70, 70, 70)

/*** Global variable ***/

/*** Function ***/
ImageProcessor::ImageProcessor()
{
    frame_cnt_ = 0;
    vanishment_y_ = 1280 / 2;
}

ImageProcessor::~ImageProcessor()
{
}

int32_t ImageProcessor::Initialize(const ImageProcessorIf::InputParam& input_param)
{
    if (object_detection_.Initialize(input_param.work_dir, input_param.num_threads) != ObjectDetection::kRetOk) {
        object_detection_.Finalize();
        return kRetErr;
    }

    if (lane_detection_.Initialize(input_param.work_dir, input_param.num_threads) != LaneDetection::kRetOk) {
        lane_detection_.Finalize();
        return kRetErr;
    }

    if (segmentation_engine_.Initialize(input_param.work_dir, input_param.num_threads) != SemanticSegmentationEngine::kRetOk) {
        segmentation_engine_.Finalize();
        return kRetErr;
    }

    if (depth_engine_.Initialize(input_param.work_dir, input_param.num_threads) != DepthEngine::kRetOk) {
        depth_engine_.Finalize();
        return kRetErr;
    }

    frame_cnt_ = 0;
    vanishment_y_ = 1280 / 2;

    return kRetOk;
}

int32_t ImageProcessor::Finalize(void)
{
    if (object_detection_.Finalize() != ObjectDetection::kRetOk) {
        return kRetErr;
    }

    if (lane_detection_.Finalize() != LaneDetection::kRetOk) {
        return kRetErr;
    }

    if (segmentation_engine_.Finalize() != SemanticSegmentationEngine::kRetOk) {
        return kRetErr;
    }

    if (depth_engine_.Finalize() != DepthEngine::kRetOk) {
        return kRetErr;
    }

    return kRetOk;
}

int32_t ImageProcessor::Command(int32_t cmd)
{
    switch (cmd) {
    case 0:
    default:
        PRINT_E("command(%d) is not supported\n", cmd);
        return kRetErr;
    }
    return kRetOk;
}

int32_t ImageProcessor::Process(const cv::Mat& mat_original, ImageProcessorIf::Result& result)
{
    /*** Initialize internal parameters using input image information ***/
    if (frame_cnt_ == 0) {
        ResetCamera(mat_original.cols, mat_original.rows);
    }

    /*** Run inference ***/
    if (object_detection_.Process(mat_original, mat_transform_, camera_real_) != ObjectDetection::kRetOk) {
        return kRetErr;
    }

    if (lane_detection_.Process(mat_original, mat_transform_) != LaneDetection::kRetOk) {
        return kRetErr;
    }

    SemanticSegmentationEngine::Result segmentation_result;
    if (segmentation_engine_.Process(mat_original, segmentation_result) != SemanticSegmentationEngine::kRetOk) {
        return kRetErr;
    }

    DepthEngine::Result depth_result;
    if (depth_engine_.Process(mat_original, depth_result) != DepthEngine::kRetOk) {
        return -1;
    }

    /*** Create Mat for output ***/
    cv::Mat mat = mat_original.clone();
    cv::Mat mat_topview;
    cv::Mat mat_depth;
    cv::Mat mat_segmentation;
    //CreateTopViewMat(mat_original, mat_topview);

    /*** Draw result ***/
    const auto& time_draw0 = std::chrono::steady_clock::now();
    DrawSegmentation(mat_segmentation, segmentation_result);
    cv::resize(mat_segmentation, mat_segmentation, mat.size());
    //cv::add(mat_segmentation, mat, mat);
    CreateTopViewMat(mat_segmentation, mat_topview);
    mat_segmentation = mat_segmentation(cv::Rect(0, vanishment_y_, mat_segmentation.cols, mat_segmentation.rows - vanishment_y_));
    cv::line(mat, cv::Point(0, camera_real_.EstimateVanishmentY()), cv::Point(mat.cols, camera_real_.EstimateVanishmentY()), cv::Scalar(0, 0, 0), 1);
    lane_detection_.Draw(mat, mat_topview);
    object_detection_.Draw(mat, mat_topview);
    DrawDepth(mat_depth, depth_result);
    const auto& time_draw1 = std::chrono::steady_clock::now();

    /*** Draw statistics ***/
    double time_draw = (time_draw1 - time_draw0).count() / 1000000.0;
    double time_inference = object_detection_.GetTimeInference() + lane_detection_.GetTimeInference() + segmentation_result.time_inference + depth_result.time_inference;
    DrawFps(mat, time_inference, time_draw, cv::Point(0, 0), 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(180, 180, 180), true);

    /*** Update internal status ***/
    frame_cnt_++;

    /*** Return the results ***/
    result.mat_output = mat;
    result.mat_output_segmentation = mat_segmentation;
    result.mat_output_depth = mat_depth;
    result.mat_output_topview = mat_topview;
    result.time_pre_process = lane_detection_.GetTimePreProcess() + lane_detection_.GetTimePreProcess() + segmentation_result.time_pre_process + depth_result.time_pre_process;
    result.time_inference = lane_detection_.GetTimeInference() + lane_detection_.GetTimeInference() + segmentation_result.time_inference + depth_result.time_inference;
    result.time_post_process = lane_detection_.GetTimePostProcess() + lane_detection_.GetTimePostProcess() + segmentation_result.time_post_process + depth_result.time_post_process;

    return kRetOk;
}


void ImageProcessor::DrawDepth(cv::Mat& mat, const DepthEngine::Result& depth_result)
{
    if (!depth_result.mat_out.empty()) {
        cv::applyColorMap(depth_result.mat_out, mat, cv::COLORMAP_PLASMA);
    }
}

void ImageProcessor::DrawSegmentation(cv::Mat& mat_segmentation, const SemanticSegmentationEngine::Result& segmentation_result)
{
    /* Draw on NormalView */
    std::vector<cv::Mat> mat_segmentation_list(4, cv::Mat());
#pragma omp parallel for
    for (int32_t i = 0; i < segmentation_result.image_list.size(); i++) {
        cv::Mat mat_fp32_3;
        cv::cvtColor(segmentation_result.image_list[i], mat_fp32_3, cv::COLOR_GRAY2BGR); /* 1channel -> 3 channel */
        cv::multiply(mat_fp32_3, GetColorForSegmentation(i), mat_fp32_3);
        mat_fp32_3.convertTo(mat_fp32_3, CV_8UC3, 1, 0);
        mat_segmentation_list[i] = mat_fp32_3;
    }

//#pragma omp parallel for  /* don't use */
    mat_segmentation = cv::Mat::zeros(mat_segmentation_list[0].size(), CV_8UC3);
    for (int32_t i = 0; i < mat_segmentation_list.size(); i++) {
        cv::add(mat_segmentation, mat_segmentation_list[i], mat_segmentation);
    }
}


void ImageProcessor::DrawFps(cv::Mat& mat, double time_inference, double time_draw, cv::Point pos, double font_scale, int32_t thickness, cv::Scalar color_front, cv::Scalar color_back, bool is_text_on_rect)
{
    char text[64];
    static auto time_previous = std::chrono::steady_clock::now();
    auto time_now = std::chrono::steady_clock::now();
    double fps = 1e9 / (time_now - time_previous).count();
    time_previous = time_now;
    snprintf(text, sizeof(text), "FPS: %.1f, Inference: %.1f [ms], Draw: %.1f [ms]", fps, time_inference, time_draw);
    CommonHelper::DrawText(mat, text, cv::Point(0, 0), 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(180, 180, 180), true);
}


cv::Scalar ImageProcessor::GetColorForSegmentation(int32_t id)
{
    switch (id) {
    default:
    case 0: /* BG */
        return COLOR_BG;
    case 1: /* road */
        return CommonHelper::CreateCvColor(255, 0, 0);
    case 2: /* curbs */
        return CommonHelper::CreateCvColor(0, 255, 0);
    case 3: /* marks */
        return CommonHelper::CreateCvColor(0, 0, 255);
    }
}

void ImageProcessor::ResetCamera(int32_t width, int32_t height, float fov_deg)
{
    if (width > 0 && height > 0 && fov_deg > 0) {
        camera_real_.parameter.SetIntrinsic(width, height, CameraModel::FocalLength(width, fov_deg));
        camera_top_.parameter.SetIntrinsic(width, height, CameraModel::FocalLength(width, fov_deg));
    }
    camera_real_.parameter.SetExtrinsic(
        { 0.0f, 0.0f, 0.0f },    /* rvec [deg] */
        { 0.0f, 1.5f, 0.0f });   /* tvec */
    camera_top_.parameter.SetExtrinsic(
        { 90.0f, 0.0f, 0.0f },    /* rvec [deg] */
        { 0.0f, 8.0f, 5.0f });   /* tvec */  /* tvec is in camera coordinate, so Z is height because pitch = 90 */
    CreateTransformMat();
    vanishment_y_ = std::max(0, std::min(height, camera_real_.EstimateVanishmentY()));
}

void ImageProcessor::GetCameraParameter(float& focal_length, std::array<float, 3>& real_rvec, std::array<float, 3>& real_tvec, std::array<float, 3>& top_rvec, std::array<float, 3>& top_tvec)
{
    focal_length = camera_real_.parameter.fx();
    camera_real_.parameter.GetExtrinsic(real_rvec, real_tvec);
    camera_top_.parameter.GetExtrinsic(top_rvec, top_tvec);
}

void ImageProcessor::SetCameraParameter(float focal_length, const std::array<float, 3>& real_rvec, const std::array<float, 3>& real_tvec, const std::array<float, 3>& top_rvec, const std::array<float, 3>& top_tvec)
{
    camera_real_.parameter.fx() = focal_length;
    camera_real_.parameter.fy() = focal_length;
    camera_top_.parameter.fx() = focal_length;
    camera_top_.parameter.fy() = focal_length;
    camera_real_.parameter.SetExtrinsic(real_rvec, real_tvec);
    camera_top_.parameter.SetExtrinsic(top_rvec, top_tvec);
    CreateTransformMat();
    vanishment_y_ = std::max(0, std::min(camera_real_.parameter.height, camera_real_.EstimateVanishmentY()));
}

void ImageProcessor::CreateTransformMat()
{
    /*** Generate mapping b/w object points (3D: world coordinate) and image points (real camera) */
    std::vector<cv::Point3f> object_point_list = {   /* Target area (possible road area) */
        { -1.0f, 0, 10.0f },
        {  1.0f, 0, 10.0f },
        { -1.0f, 0,  3.0f },
        {  1.0f, 0,  3.0f },
    };
    std::vector<cv::Point2f> image_point_real_list;
    cv::projectPoints(object_point_list, camera_real_.parameter.rvec, camera_real_.parameter.tvec, camera_real_.parameter.K, camera_real_.parameter.dist_coeff, image_point_real_list);

    /* Convert to image points (2D) using the top view camera (virtual camera) */
    std::vector<cv::Point2f> image_point_top_list;
    cv::projectPoints(object_point_list, camera_top_.parameter.rvec, camera_top_.parameter.tvec, camera_top_.parameter.K, camera_top_.parameter.dist_coeff, image_point_top_list);

    mat_transform_ = cv::getPerspectiveTransform(&image_point_real_list[0], &image_point_top_list[0]);
}

void ImageProcessor::CreateTopViewMat(const cv::Mat& mat_original, cv::Mat& mat_topview)
{
    /* Perspective Transform */   
    mat_topview = cv::Mat(mat_original.size(), CV_8UC3, COLOR_BG);
    cv::warpPerspective(mat_original, mat_topview, mat_transform_, mat_topview.size(), cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
    //mat_topview = mat_topview(cv::Rect(0, 360, 1280, 360));
}
