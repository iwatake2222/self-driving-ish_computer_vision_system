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
#include "bounding_box.h"
#include "detection_engine.h"
#include "tracker.h"

#include "image_processor_if.h"
#include "image_processor.h"


/*** Macro ***/
#define TAG "ImageProcessor"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/*** Global variable ***/

/*** Function ***/
ImageProcessor::ImageProcessor()
{
    frame_cnt = 0;
}

ImageProcessor::~ImageProcessor()
{
}

int32_t ImageProcessor::Initialize(const ImageProcessorIf::InputParam& input_param)
{
    if (m_detection_engine.Initialize(input_param.work_dir, input_param.num_threads) != DetectionEngine::kRetOk) {
        m_detection_engine.Finalize();
        return kRetErr;
    }

    frame_cnt = 0;

    return kRetOk;
}

int32_t ImageProcessor::Finalize(void)
{
    if (m_detection_engine.Finalize() != DetectionEngine::kRetOk) {
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
    /*** Create Mat for output ***/
    cv::Mat mat = mat_original.clone();

    if (frame_cnt == 0) {
        ResetCamera(mat_original.cols, mat_original.rows, 130.0f);
    }
    cv::Mat mat_topview;
    CreateTopViewMat(mat_original, mat_topview);
    

    /*** Semantic Segmentation ***/
    /* todo */

    /*** Lane Detection ***/
    /* todo */

    /*** Object Detection ***/
    DetectionEngine::Result det_result;
    ProcessObjectDetection(mat_original, det_result);
    DrawObjectDetection(mat, mat_topview, det_result);

    /*** Draw statistics ***/
    CommonHelper::DrawText(mat, "DET: " + std::to_string(det_result.bbox_list.size()) + ", TRACK: " + std::to_string(m_tracker.GetTrackList().size()), cv::Point(0, 20), 0.7, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(220, 220, 220));
    DrawFps(mat, det_result.time_inference, cv::Point(0, 0), 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(180, 180, 180), true);

    /*** Update internal status ***/
    frame_cnt++;

    /*** Return the results ***/
    result.mat_output = mat;
    result.mat_output_topview = mat_topview;
    result.time_pre_process = det_result.time_pre_process;
    result.time_inference = det_result.time_inference;
    result.time_post_process = det_result.time_post_process;

    return kRetOk;
}

void ImageProcessor::DrawObjectDetection(cv::Mat& mat, cv::Mat& mat_topview, const DetectionEngine::Result& det_result)
{
    /* Draw on NormalView */
    cv::rectangle(mat, cv::Rect(det_result.crop.x, det_result.crop.y, det_result.crop.w, det_result.crop.h), CommonHelper::CreateCvColor(0, 0, 0), 2);
    for (const auto& bbox : det_result.bbox_list) {
        cv::rectangle(mat, cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h), CommonHelper::CreateCvColor(0, 0, 0), 1);
    }

    auto& track_list = m_tracker.GetTrackList();
    for (auto& track : track_list) {
        if (track.GetDetectedCount() < 2) continue;
        const auto& bbox = track.GetLatestData().bbox;
        /* Use white rectangle for the object which was not detected but just predicted */
        cv::Scalar color = bbox.score == 0 ? CommonHelper::CreateCvColor(255, 255, 255) : GetColorForId(track.GetId());
        cv::rectangle(mat, cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h), color, 2);
        CommonHelper::DrawText(mat, std::to_string(track.GetId()) + ": " + bbox.label, cv::Point(bbox.x, bbox.y - 13), 0.35, 1, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(220, 220, 220));

        auto& track_history = track.GetDataHistory();
        for (size_t i = 1; i < track_history.size(); i++) {
            cv::Point p0(track_history[i].bbox.x + track_history[i].bbox.w / 2, track_history[i].bbox.y + track_history[i].bbox.h);
            cv::Point p1(track_history[i - 1].bbox.x + track_history[i - 1].bbox.w / 2, track_history[i - 1].bbox.y + track_history[i - 1].bbox.h);
            cv::line(mat, p0, p1, GetColorForId(track.GetId()));
        }
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
        cv::circle(mat_topview, topview_points[i], 5, GetColorForId(track.GetId()), -1);

        CommonHelper::DrawText(mat_topview, track_data.bbox.label, topview_points[i], 0.35, 1, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(220, 220, 220));

        //auto& track_history = track.GetDataHistory();
        //for (size_t i = 1; i < track_history.size(); i++) {
        //    cv::Point p0(track_history[i - 1].topview.x, track_history[i - 1].topview.y);
        //    cv::Point p1(track_history[i].topview.x, track_history[i].topview.y);
        //    cv::line(mat_topview, p0, p1, GetColorForId(track.GetId()));
        //}
    }
}

int32_t ImageProcessor::ProcessObjectDetection(const cv::Mat& mat_original, DetectionEngine::Result& det_result)
{
    if (m_detection_engine.Process(mat_original, det_result) != DetectionEngine::kRetOk) {
        return kRetErr;
    }
    m_tracker.Update(det_result.bbox_list);
    return kRetOk;
}

void ImageProcessor::DrawFps(cv::Mat& mat, double time_inference, cv::Point pos, double font_scale, int32_t thickness, cv::Scalar color_front, cv::Scalar color_back, bool is_text_on_rect)
{
    char text[64];
    static auto time_previous = std::chrono::steady_clock::now();
    auto time_now = std::chrono::steady_clock::now();
    double fps = 1e9 / (time_now - time_previous).count();
    time_previous = time_now;
    snprintf(text, sizeof(text), "FPS: %.1f, Inference: %.1f [ms]", fps, time_inference);
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

void ImageProcessor::ResetCamera(int32_t width, int32_t height, float fov_deg)
{
    camera_real.parameter.SetIntrinsic(width, height, CameraModel::FocalLength(width, fov_deg));
    camera_top.parameter.SetIntrinsic(width, height, CameraModel::FocalLength(width, fov_deg));
    camera_real.parameter.SetExtrinsic(
        { 0.0f, 0.0f, 0.0f },    /* rvec [deg] */
        { 0.0f, 1.0f, 0.0f });   /* tvec */
    camera_top.parameter.SetExtrinsic(
        { 90.0f, 0.0f, 0.0f },    /* rvec [deg] */
        { 0.0f, 8.0f, 7.0f });   /* tvec */  /* tvec is in camera coordinate, so Z is height because pitch = 90 */
    CreateTransformMat();
}

void ImageProcessor::GetCameraParameter(std::array<float, 3>& real_rvec, std::array<float, 3>& real_tvec, std::array<float, 3>& top_rvec, std::array<float, 3>& top_tvec)
{
    camera_real.parameter.GetExtrinsic(real_rvec, real_tvec);
    camera_top.parameter.GetExtrinsic(top_rvec, top_tvec);
}

void ImageProcessor::SetCameraParameter(const std::array<float, 3>& real_rvec, const std::array<float, 3>& real_tvec, const std::array<float, 3>& top_rvec, const std::array<float, 3>& top_tvec)
{
    camera_real.parameter.SetExtrinsic(real_rvec, real_tvec);
    camera_top.parameter.SetExtrinsic(top_rvec, top_tvec);
    CreateTransformMat();
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
    cv::projectPoints(object_point_list, camera_real.parameter.rvec, camera_real.parameter.tvec, camera_real.parameter.K, camera_real.parameter.dist_coeff, image_point_real_list);

    /* Convert to image points (2D) using the top view camera (virtual camera) */
    std::vector<cv::Point2f> image_point_top_list;
    cv::projectPoints(object_point_list, camera_top.parameter.rvec, camera_top.parameter.tvec, camera_top.parameter.K, camera_top.parameter.dist_coeff, image_point_top_list);

    mat_transform_ = cv::getPerspectiveTransform(&image_point_real_list[0], &image_point_top_list[0]);
}

void ImageProcessor::CreateTopViewMat(const cv::Mat& mat_original, cv::Mat& mat_topview)
{
    /* Perspective Transform */   
    mat_topview = cv::Mat(mat_original.size(), CV_8UC3, cv::Scalar(70, 70, 70));
    cv::warpPerspective(mat_original, mat_topview, mat_transform_, mat_topview.size(), cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
}
