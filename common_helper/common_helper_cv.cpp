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

/* for OpenCV */
#include <opencv2/opencv.hpp>

#include "common_helper.h"
#include "common_helper_cv.h"


cv::Scalar CommonHelper::CreateCvColor(int32_t b, int32_t g, int32_t r)
{
#ifdef CV_COLOR_IS_RGB
    return cv::Scalar(r, g, b);
#else
    return cv::Scalar(b, g, r);
#endif
}

void CommonHelper::DrawText(cv::Mat& mat, const std::string& text, cv::Point pos, double font_scale, int32_t thickness, cv::Scalar color_front, cv::Scalar color_back, bool is_text_on_rect)
{
    int32_t baseline = 0;
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
    baseline += thickness;
    pos.y += textSize.height;
    if (is_text_on_rect) {
        cv::rectangle(mat, pos + cv::Point(0, baseline), pos + cv::Point(textSize.width, -textSize.height), color_back, -1);
        cv::putText(mat, text, pos, cv::FONT_HERSHEY_SIMPLEX, font_scale, color_front, thickness);
    } else {
        cv::putText(mat, text, pos, cv::FONT_HERSHEY_SIMPLEX, font_scale, color_back, thickness * 3);
        cv::putText(mat, text, pos, cv::FONT_HERSHEY_SIMPLEX, font_scale, color_front, thickness);
    }
}

void CommonHelper::CropResizeCvt(const cv::Mat& org, cv::Mat& dst, int32_t& crop_x, int32_t& crop_y, int32_t& crop_w, int32_t& crop_h, bool is_rgb, int32_t crop_type, bool resize_by_linear)
{
    const int32_t interpolation_flag = resize_by_linear ? cv::INTER_LINEAR : cv::INTER_NEAREST;

    cv::Mat src = org(cv::Rect(crop_x, crop_y, crop_w, crop_h));

    if (crop_type == kCropTypeStretch) {
        cv::resize(src, dst, dst.size(), 0, 0, interpolation_flag);
    } else if (crop_type == kCropTypeCut) {
        float aspect_ratio_src = static_cast<float>(src.cols) / src.rows;
        float aspect_ratio_dst = static_cast<float>(dst.cols) / dst.rows;
        cv::Rect target_rect(0, 0, src.cols, src.rows);
        if (aspect_ratio_src > aspect_ratio_dst) {
            target_rect.width = static_cast<int32_t>(src.rows * aspect_ratio_dst);
            target_rect.x = (src.cols - target_rect.width) / 2;
        } else {
            target_rect.height = static_cast<int32_t>(src.cols / aspect_ratio_dst);
            target_rect.y = (src.rows - target_rect.height) / 2;
        }
        cv::Mat target = src(target_rect);
        cv::resize(target, dst, dst.size(), 0, 0, interpolation_flag);
        crop_x += target_rect.x;
        crop_y += target_rect.y;
        crop_w = target_rect.width;
        crop_h = target_rect.height;
    } else {
        float aspect_ratio_src = static_cast<float>(src.cols) / src.rows;
        float aspect_ratio_dst = static_cast<float>(dst.cols) / dst.rows;
        cv::Rect target_rect(0, 0, dst.cols, dst.rows);
        if (aspect_ratio_src > aspect_ratio_dst) {
            target_rect.height = static_cast<int32_t>(target_rect.width / aspect_ratio_src);
            target_rect.y = (dst.rows - target_rect.height) / 2;
        } else {
            target_rect.width = static_cast<int32_t>(target_rect.height * aspect_ratio_src);
            target_rect.x = (dst.cols - target_rect.width) / 2;
        }
        cv::Mat target = dst(target_rect);
        cv::resize(src, target, target.size(), 0, 0, interpolation_flag);
        crop_x -= target_rect.x * crop_w / target_rect.width;
        crop_y -= target_rect.y * crop_h / target_rect.height;
        crop_w = dst.cols * crop_w / target_rect.width;
        crop_h = dst.rows * crop_h / target_rect.height;
    }

#ifdef CV_COLOR_IS_RGB
    if (!is_rgb) {
        cv::cvtColor(dst, dst, cv::COLOR_RGB2BGR);
    }
#else
    if (is_rgb) {
        cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);
    }
#endif

}

/* https://github.com/JetsonHacksNano/CSI-Camera/blob/master/simple_camera.cpp */
/* modified by iwatake2222 */
std::string CommonHelper::CreateGStreamerPipeline(int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
        std::to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(framerate) +
        "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
        std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=True";
}

bool CommonHelper::FindSourceImage(const std::string& input_name, cv::VideoCapture& cap, int32_t width, int32_t height)
{
    if (input_name.find(".mp4") != std::string::npos || input_name.find(".avi") != std::string::npos || input_name.find(".webm") != std::string::npos) {
        cap = cv::VideoCapture(input_name);
        if (!cap.isOpened()) {
            printf("Invalid input source: %s\n", input_name.c_str());
            return false;
        }
    } else if (input_name.find(".jpg") != std::string::npos || input_name.find(".png") != std::string::npos || input_name.find(".bmp") != std::string::npos) {
        if (cv::imread(input_name).empty()) {
            printf("Invalid input source: %s\n", input_name.c_str());
            return false;
        }
    } else {
        if (input_name == "jetson") {
            cap = cv::VideoCapture(CreateGStreamerPipeline(width, height, width, height, 60, 2));
        } else {
            int32_t cam_id = -1;
            try {
                cam_id = std::stoi(input_name);
            }
            catch (...) {}
            cap = (cam_id >= 0) ? cv::VideoCapture(cam_id) : cv::VideoCapture(input_name);
            cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
            cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        }
        if (!cap.isOpened()) {
            printf("Unable to open camera: %s\n", input_name.c_str());
            return false;
        }
    }
    return true;
}

bool CommonHelper::InputKeyCommand(cv::VideoCapture& cap)
{
    bool ret_to_quit = false;
    static bool is_pause = false;
    bool is_process_one_frame = false;
    do {
        int32_t key = cv::waitKey(1) & 0xff;
        switch (key) {
        case 'q':
            cap.release();
            ret_to_quit = true;
            break;
        case 'p':
            is_pause = !is_pause;
            break;
        case '>':
            if (is_pause) {
                is_process_one_frame = true;
            } else {
                int32_t current_frame = static_cast<int32_t>(cap.get(cv::CAP_PROP_POS_FRAMES));
                cap.set(cv::CAP_PROP_POS_FRAMES, current_frame + 100);
            }
            break;
        case '<':
            int32_t current_frame = static_cast<int32_t>(cap.get(cv::CAP_PROP_POS_FRAMES));
            if (is_pause) {
                is_process_one_frame = true;
                cap.set(cv::CAP_PROP_POS_FRAMES, current_frame - 2);
            } else {
                cap.set(cv::CAP_PROP_POS_FRAMES, current_frame - 100);
            }
            break;
        }
    } while (is_pause && !is_process_one_frame);

    return ret_to_quit;
}
