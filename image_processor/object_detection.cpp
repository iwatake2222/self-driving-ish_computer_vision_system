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
#include "object_detection.h"


/*** Macro ***/
#define TAG "ObjectDetection"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)


/*** Function ***/
int32_t ObjectDetection::Initialize(const std::string& work_dir, const int32_t num_threads)
{
    if (detection_engine_.Initialize(work_dir, num_threads) != DetectionEngine::kRetOk) {
        detection_engine_.Finalize();
        return kRetErr;
    }
    return kRetOk;
}

int32_t ObjectDetection::Finalize()
{
    if (detection_engine_.Finalize() != DetectionEngine::kRetOk) {
        return kRetErr;
    }
    return kRetOk;
}

int32_t ObjectDetection::Process(const cv::Mat& mat, const cv::Mat& mat_transform, CameraModel& camera)
{
    /* Run inference to detect objects */
    DetectionEngine::Result det_result;
    if (detection_engine_.Process(mat, det_result) != DetectionEngine::kRetOk) {
        return kRetErr;
    }
    roi_ = cv::Rect(det_result.crop.x, det_result.crop.y, det_result.crop.w, det_result.crop.h);
    time_pre_process_ = det_result.time_pre_process;
    time_inference_ = det_result.time_inference;
    time_post_process_ = det_result.time_post_process;

    /* Track */
    tracker_.Update(det_result.bbox_list);
    auto& track_list = tracker_.GetTrackList();

    /* Convert points from normal view -> top view. Store the results into track data */
    std::vector<cv::Point2f> normal_points;
    std::vector<cv::Point2f> topview_points;
    for (auto& track : track_list) {
        const auto& bbox = track.GetLatestData().bbox;
        normal_points.push_back({ bbox.x + bbox.w / 2.0f, bbox.y + bbox.h + 0.0f });
    }
    if (normal_points.size() > 0) {
        cv::perspectiveTransform(normal_points, topview_points, mat_transform);
        for (int32_t i = 0; i < static_cast<int32_t>(track_list.size()); i++) {
            auto& track_data = track_list[i].GetLatestData();
            track_data.topview_point.x = static_cast<int32_t>(topview_points[i].x);
            track_data.topview_point.y = static_cast<int32_t>(topview_points[i].y);
        }
    }

    /* Calcualte points in world coordinate (distance on ground plane) */
    int32_t vanishment_y = camera.EstimateVanishmentY();
    for (auto& track : track_list) {
        const auto& bbox = track.GetLatestData().bbox;
        auto& object_point_track = track.GetLatestData().object_point;
        cv::Point3f object_point;
        camera.ProjectImage2GroundPlane(cv::Point2f(bbox.x + bbox.w / 2.0f, bbox.y + bbox.h + 0.0f), object_point);
        if (bbox.y + bbox.h < vanishment_y) object_point.z = 999;
        object_point_track.x = object_point.x;
        object_point_track.y = object_point.y;
        object_point_track.z = object_point.z;
    }

    return kRetOk;
}

void ObjectDetection::Draw(cv::Mat& mat, cv::Mat& mat_topview)
{
    auto& track_list = tracker_.GetTrackList();

    /* Draw on NormalView */
    cv::rectangle(mat, roi_, CommonHelper::CreateCvColor(0, 0, 0), 2);
    int32_t det_num = 0;
    for (auto& track : track_list) {
        if (track.GetUndetectedCount() > 0) continue;
        const auto& bbox = track.GetLatestData().bbox_raw;
        cv::rectangle(mat, cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h), CommonHelper::CreateCvColor(0, 0, 0), 1);
        det_num++;
    }
    CommonHelper::DrawText(mat, "DET: " + std::to_string(det_num) + ", TRACK: " + std::to_string(tracker_.GetTrackList().size()), cv::Point(0, 20), 0.7, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(220, 220, 220));

    for (auto& track : track_list) {
        if (track.GetDetectedCount() < 2) continue;
        const auto& bbox = track.GetLatestData().bbox;
        const auto& object_point = track.GetLatestData().object_point;
        /* Use white rectangle for the object which was not detected but just predicted */
        cv::Scalar color = bbox.score == 0 ? CommonHelper::CreateCvColor(255, 255, 255) : GetColorForId(track.GetId());
        cv::rectangle(mat, cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h), color, 2);

        char text[32];
        snprintf(text, sizeof(text), "%.1f,%.1f", object_point.x, object_point.z);
        CommonHelper::DrawText(mat, text, cv::Point(bbox.x, bbox.y - 13), 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(220, 220, 220));

        //auto& track_history = track.GetDataHistory();
        //for (int32_t i = 1; i < static_cast<int32_t>(track_history.size()); i++) {
        //    cv::Point p0(track_history[i].bbox.x + track_history[i].bbox.w / 2, track_history[i].bbox.y + track_history[i].bbox.h);
        //    cv::Point p1(track_history[i - 1].bbox.x + track_history[i - 1].bbox.w / 2, track_history[i - 1].bbox.y + track_history[i - 1].bbox.h);
        //    cv::line(mat, p0, p1, GetColorForId(track.GetId()));
        //}
    }

    /* Draw on TopView*/
    for (int32_t i = 0; i < static_cast<int32_t>(track_list.size()); i++) {
        auto& track = track_list[i];
        const auto& bbox = track.GetLatestData().bbox;
        const auto& object_point = track.GetLatestData().object_point;
        const auto& topview_point = track.GetLatestData().topview_point;
        cv::Point p = cv::Point(topview_point.x, topview_point.y);

        if (track.GetDetectedCount() < 2) continue;
        cv::circle(mat_topview, p, 10, GetColorForId(track.GetId()), -1);
        cv::circle(mat_topview, p, 10, cv::Scalar(0, 0, 0), 2);

        char text[32];
        snprintf(text, sizeof(text), "%s", bbox.label.c_str());
        CommonHelper::DrawText(mat_topview, text, p, 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(255, 255, 255), false);
        snprintf(text, sizeof(text), "%.1f,%.1f", object_point.x, object_point.z);
        CommonHelper::DrawText(mat_topview, text, p + cv::Point(0, 15), 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(255, 255, 255), false);

        //auto& track_history = track.GetDataHistory();
        //for (int32_t i = 1; i < static_cast<int32_t>(track_history.size()); i++) {
        //    cv::Point p0(track_history[i - 1].topview_point.x, track_history[i - 1].topview_point.y);
        //    cv::Point p1(track_history[i].topview_point.x, track_history[i].topview_point.y);
        //    cv::line(mat_topview, p0, p1, GetColorForId(track.GetId()));
        //}
    }
}

cv::Scalar ObjectDetection::GetColorForId(int32_t id)
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
