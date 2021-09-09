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
#include "lane_information.h"
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
    if (detection_engine_.Initialize(input_param.work_dir, input_param.num_threads) != DetectionEngine::kRetOk) {
        detection_engine_.Finalize();
        return kRetErr;
    }

    if (lane_information_.Initialize(input_param.work_dir, input_param.num_threads) != LaneInformation::kRetOk) {
        lane_information_.Finalize();
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
    if (detection_engine_.Finalize() != DetectionEngine::kRetOk) {
        return kRetErr;
    }

    if (lane_information_.Finalize() != LaneInformation::kRetOk) {
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
    DetectionEngine::Result det_result;
    if (detection_engine_.Process(mat_original, det_result) != DetectionEngine::kRetOk) {
        return kRetErr;
    }
    tracker_.Update(det_result.bbox_list);

    if (lane_information_.Process(mat_original, mat_transform_) != LaneInformation::kRetOk) {
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
    cv::Mat mat_depth;
    cv::Mat mat_segmentation;
    cv::Mat mat_topview;
    //CreateTopViewMat(mat_original, mat_topview);

    /*** Draw result ***/
    const auto& time_draw0 = std::chrono::steady_clock::now();
    DrawSegmentation(mat_segmentation, segmentation_result);
    cv::resize(mat_segmentation, mat_segmentation, mat.size());
    //cv::add(mat_segmentation, mat, mat);
    CreateTopViewMat(mat_segmentation, mat_topview);
    mat_segmentation = mat_segmentation(cv::Rect(0, vanishment_y_, mat_segmentation.cols, mat_segmentation.rows - vanishment_y_));

    lane_information_.Draw(mat, mat_topview);
    DrawObjectDetection(mat, mat_topview, det_result);
    DrawDepth(mat_depth, depth_result);
    const auto& time_draw1 = std::chrono::steady_clock::now();

    /*** Draw statistics ***/
    double time_draw = (time_draw1 - time_draw0).count() / 1000000.0;
    double time_inference = det_result.time_inference + lane_information_.GetTimeInference() + segmentation_result.time_inference + depth_result.time_inference;
    CommonHelper::DrawText(mat, "DET: " + std::to_string(det_result.bbox_list.size()) + ", TRACK: " + std::to_string(tracker_.GetTrackList().size()), cv::Point(0, 20), 0.7, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(220, 220, 220));
    DrawFps(mat, time_inference, time_draw, cv::Point(0, 0), 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(180, 180, 180), true);
    cv::line(mat, cv::Point(0, camera_real_.EstimateVanishmentY()), cv::Point(mat.cols, camera_real_.EstimateVanishmentY()), cv::Scalar(0, 0, 0), 1);

    /*** Update internal status ***/
    frame_cnt_++;

    /*** Return the results ***/
    result.mat_output = mat;
    result.mat_output_segmentation = mat_segmentation;
    result.mat_output_depth = mat_depth;
    result.mat_output_topview = mat_topview;
    result.time_pre_process = det_result.time_pre_process + lane_information_.GetTimePreProcess() + segmentation_result.time_pre_process + depth_result.time_pre_process;
    result.time_inference = det_result.time_inference + lane_information_.GetTimeInference() + segmentation_result.time_inference + depth_result.time_inference;
    result.time_post_process = det_result.time_post_process + lane_information_.GetTimePostProcess() + segmentation_result.time_post_process + depth_result.time_post_process;

    return kRetOk;
}


void ImageProcessor::DrawDepth(cv::Mat& mat, const DepthEngine::Result& depth_result)
{
    float scale_w = static_cast<float>(depth_result.mat_out.cols) / depth_result.crop.w;
    float scale_h = static_cast<float>(depth_result.mat_out.rows) / depth_result.crop.h;
    cv::Rect crop = cv::Rect(depth_result.crop.x, depth_result.crop.y, depth_result.crop.w, depth_result.crop.h);
    if (crop.x < 0) {
        crop.x *= -1;
        crop.width -= 2 * crop.x;
    }
    if (crop.y < 0) {
        crop.y *= -1;
        crop.height -= 2 * crop.y;
    }
    crop.x = static_cast<int32_t>(crop.x * scale_w);
    crop.width = static_cast<int32_t>(crop.width * scale_w);
    crop.y = static_cast<int32_t>(crop.y * scale_h);
    crop.height = static_cast<int32_t>(crop.height * scale_h);
    cv::applyColorMap(depth_result.mat_out(crop), mat, cv::COLORMAP_JET);
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

void ImageProcessor::DrawObjectDetection(cv::Mat& mat, cv::Mat& mat_topview, const DetectionEngine::Result& det_result)
{
    /* Draw on NormalView */
    cv::rectangle(mat, cv::Rect(det_result.crop.x, det_result.crop.y, det_result.crop.w, det_result.crop.h), CommonHelper::CreateCvColor(0, 0, 0), 2);
    for (const auto& bbox : det_result.bbox_list) {
        cv::rectangle(mat, cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h), CommonHelper::CreateCvColor(0, 0, 0), 1);
    }

    auto& track_list = tracker_.GetTrackList();
    for (auto& track : track_list) {
        if (track.GetDetectedCount() < 2) continue;
        const auto& bbox = track.GetLatestData().bbox;
        /* Use white rectangle for the object which was not detected but just predicted */
        cv::Scalar color = bbox.score == 0 ? CommonHelper::CreateCvColor(255, 255, 255) : GetColorForId(track.GetId());
        cv::rectangle(mat, cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h), color, 2);

        cv::Point3f object_point;
        camera_real_.ProjectImage2GroundPlane(cv::Point2f(bbox.x + bbox.w / 2.0f, bbox.y + bbox.h + 0.0f), object_point);
        if (bbox.y + bbox.h + 0.0f < vanishment_y_) object_point.z = 999;
        char text[32];
        snprintf(text, sizeof(text), "%s:%.1f,%.1f[m]", bbox.label.c_str(), object_point.x, object_point.z);
        CommonHelper::DrawText(mat, text, cv::Point(bbox.x, bbox.y - 13), 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(220, 220, 220));

        //auto& track_history = track.GetDataHistory();
        //for (size_t i = 1; i < track_history.size(); i++) {
        //    cv::Point p0(track_history[i].bbox.x + track_history[i].bbox.w / 2, track_history[i].bbox.y + track_history[i].bbox.h);
        //    cv::Point p1(track_history[i - 1].bbox.x + track_history[i - 1].bbox.w / 2, track_history[i - 1].bbox.y + track_history[i - 1].bbox.h);
        //    cv::line(mat, p0, p1, GetColorForId(track.GetId()));
        //}
    }

    /* Draw on TopView*/
    std::vector<cv::Point2f> normal_points;
    std::vector<cv::Point2f> topview_points;
    for (auto& track : track_list) {
        const auto& bbox = track.GetLatestData().bbox;
        normal_points.push_back({ bbox.x + bbox.w / 2.0f, bbox.y + bbox.h + 0.0f });
    }
    if (normal_points.size() > 0) {
        cv::perspectiveTransform(normal_points, topview_points, mat_transform_);
    }

    for (int32_t i = 0; i < track_list.size(); i++) {
        auto& track = track_list[i];
        auto& track_data = track.GetLatestData();
        track_data.topview.x = static_cast<int32_t>(topview_points[i].x);
        track_data.topview.y = static_cast<int32_t>(topview_points[i].y);
        
        if (track.GetDetectedCount() < 2) continue;
        cv::circle(mat_topview, topview_points[i], 10, GetColorForId(track.GetId()), -1);
        cv::circle(mat_topview, topview_points[i], 10, cv::Scalar(0, 0, 0), 2);

        cv::Point3f object_point;
        camera_real_.ProjectImage2GroundPlane({track_data.bbox.x + track_data.bbox.w / 2.0f, track_data.bbox.y + track_data.bbox.h + 0.0f}, object_point);
        char text[32];
        snprintf(text, sizeof(text), "%s:%.1f,%.1f[m]", track_data.bbox.label.c_str(), object_point.x, object_point.z);
        CommonHelper::DrawText(mat_topview, text, topview_points[i], 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(220, 220, 220));

        //auto& track_history = track.GetDataHistory();
        //for (size_t i = 1; i < track_history.size(); i++) {
        //    cv::Point p0(track_history[i - 1].topview.x, track_history[i - 1].topview.y);
        //    cv::Point p1(track_history[i].topview.x, track_history[i].topview.y);
        //    cv::line(mat_topview, p0, p1, GetColorForId(track.GetId()));
        //}
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

cv::Scalar ImageProcessor::GetColorForId(int32_t id)
{
    static constexpr int32_t kMaxNum = 100;
    static std::vector<cv::Scalar> color_list;
    if (color_list.empty()) {
        std::srand(123);
        for (int32_t i = 0; i < kMaxNum; i++) {
            color_list.push_back(CommonHelper::CreateCvColor(std::rand() % 255, std::rand() % 255, std::rand() % 255));
        }
    }
    return color_list[id % kMaxNum];
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
        { 0.0f, 1.0f, 0.0f });   /* tvec */
    camera_top_.parameter.SetExtrinsic(
        { 90.0f, 0.0f, 0.0f },    /* rvec [deg] */
        { 0.0f, 8.0f, 7.0f });   /* tvec */  /* tvec is in camera coordinate, so Z is height because pitch = 90 */
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
