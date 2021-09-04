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
#include <string>
#include <algorithm>
#include <chrono>

/* for OpenCV */
#include <opencv2/opencv.hpp>

#define CVUI_IMPLEMENTATION
#include "cvui.h"

/* for My modules */
#include "image_processor_if.h"
#include "common_helper_cv.h"

/*** Macro ***/
#define WORK_DIR                      RESOURCE_DIR
#define DEFAULT_INPUT_IMAGE           RESOURCE_DIR"/dashcam_00.jpg"
#define LOOP_NUM_FOR_TIME_MEASUREMENT 10

static constexpr char kWindowNormal[] = "WindowNormal";
static constexpr char kWindowTopView[] = "WindowTopView";
static constexpr char kWindowParam[] = "WindowParam";
static constexpr float kFovDeg = 130.0f;


/*** Global variable ***/
/* variables for processing time measurement */
static double total_time_all = 0;
static double total_time_cap = 0;
static double total_time_image_process = 0;
static bool is_pause = false;
static bool is_process_one_frame = false;

/*** Function ***/
static int32_t loop_main(std::unique_ptr<ImageProcessorIf>& image_processor, int32_t frame_cnt, std::string& input_name, cv::VideoCapture& cap, cv::VideoWriter& writer)
{
    const auto& time_all0 = std::chrono::steady_clock::now();
    /* Read image */
    const auto& time_cap0 = std::chrono::steady_clock::now();
    static cv::Mat mat_original;   /* to reuse image in the previous frame during pose*/
    if (cap.isOpened()) {
        if (!is_pause || is_process_one_frame) {
            cap.read(mat_original);
        }
    } else {
        mat_original = cv::imread(input_name);
    }
    if (mat_original.empty()) return -1;
    const auto& time_cap1 = std::chrono::steady_clock::now();

    /* Call image processor library */
    const auto& time_image_process0 = std::chrono::steady_clock::now();
    ImageProcessorIf::Result result;
    image_processor->Process(mat_original, result);
    const auto& time_image_process1 = std::chrono::steady_clock::now();

    /* Display result */
    if (writer.isOpened()) writer.write(result.mat_output);
    cvui::imshow(kWindowNormal, result.mat_output);
    cvui::imshow(kWindowTopView, result.mat_output_topview);

    /* Print processing time */
    const auto& time_all1 = std::chrono::steady_clock::now();
    double time_all = (time_all1 - time_all0).count() / 1000000.0;
    double time_cap = (time_cap1 - time_cap0).count() / 1000000.0;
    double time_image_process = (time_image_process1 - time_image_process0).count() / 1000000.0;
    printf("Total:               %9.3lf [msec]\n", time_all);
    printf("  Capture:           %9.3lf [msec]\n", time_cap);
    printf("  Image processing:  %9.3lf [msec]\n", time_image_process);
    printf("    Pre processing:  %9.3lf [msec]\n", result.time_pre_process);
    printf("    Inference:       %9.3lf [msec]\n", result.time_inference);
    printf("    Post processing: %9.3lf [msec]\n", result.time_post_process);
    printf("=== Finished %d frame ===\n\n", frame_cnt);

    if (frame_cnt > 0) {    /* do not count the first process because it may include initialize process */
        total_time_all += time_all;
        total_time_cap += time_cap;
        total_time_image_process += time_image_process;
    }
    return 0;
}




#define MAKE_GUI_SETTING_FLOAT(VAL, LABEL, STEP, FORMAT, RANGE0, RANGE1) {\
cvui::beginColumn(-1, -1, 2);\
double temp_double_current = static_cast<double>(VAL);\
double temp_double_new = temp_double_current;\
float temp_float_current = VAL;\
float temp_float_new = temp_float_current;\
cvui::text(LABEL);\
cvui::counter(&temp_double_new, STEP, FORMAT);\
cvui::trackbar<float>(200, &temp_float_new, RANGE0, RANGE1);\
if (temp_double_new != temp_double_current) VAL = static_cast<float>(temp_double_new);\
if (temp_float_new != temp_float_current) VAL = temp_float_new;\
cvui::endColumn();\
}

static void loop_param(std::unique_ptr<ImageProcessorIf>& image_processor)
{
    cvui::context(kWindowParam);
    cv::Mat mat = cv::Mat(800, 300, CV_8UC3, cv::Scalar(70, 70, 70));

    cvui::beginColumn(mat, 10, 10, -1, -1, 2);
    {
        if (cvui::button(120, 20, "Reset")) {
            image_processor->ResetCamera();
        }

        float f;
        std::array<float, 3> real_rvec;
        std::array<float, 3> real_tvec;
        std::array<float, 3> top_rvec;
        std::array<float, 3> top_tvec;
        image_processor->GetCameraParameter(f, real_rvec, real_tvec, top_rvec, top_tvec);

        cvui::text("Camera Parameter (Intrinsic)");     
        MAKE_GUI_SETTING_FLOAT(f, "Focal Length", 10.0f, "%.0Lf", 0.0f, 1000.0f);

        cvui::text("Top Camera Parameter (Extrinsic)");
        MAKE_GUI_SETTING_FLOAT(top_rvec[0], "Pitch", 1.0f, "%.0Lf", -90.0f, 90.0f);
        MAKE_GUI_SETTING_FLOAT(top_rvec[1], "Yaw", 1.0f, "%.0Lf", -90.0f, 90.0f);
        MAKE_GUI_SETTING_FLOAT(top_rvec[2], "Roll", 1.0f, "%.0Lf", -90.0f, 90.0f);

        cvui::text("Real Camera Parameter (Extrinsic)");
        MAKE_GUI_SETTING_FLOAT(real_tvec[1], "Height", 1.0f, "%.0Lf", 0.0f, 5.0f);

        MAKE_GUI_SETTING_FLOAT(real_rvec[0], "Pitch", 1.0f, "%.0Lf", -90.0f, 90.0f);
        MAKE_GUI_SETTING_FLOAT(real_rvec[1], "Yaw", 1.0f, "%.0Lf", -90.0f, 90.0f);
        MAKE_GUI_SETTING_FLOAT(real_rvec[2], "Roll", 1.0f, "%.0Lf", -90.0f, 90.0f);

        image_processor->SetCameraParameter(f, real_rvec, real_tvec, top_rvec, top_tvec);

    }
    cvui::endColumn();

    cvui::imshow(kWindowParam, mat);
}

static void CallbackMouseMain(int32_t event, int32_t x, int32_t y, int32_t flags, void* userdata)
{
    ImageProcessorIf* image_processor = (ImageProcessorIf*)(userdata);
    float f;
    std::array<float, 3> real_rvec;
    std::array<float, 3> real_tvec;
    std::array<float, 3> top_rvec;
    std::array<float, 3> top_tvec;
    image_processor->GetCameraParameter(f, real_rvec, real_tvec, top_rvec, top_tvec);

    static constexpr float kIncAnglePerPx = 0.1f;
    static constexpr int32_t kInvalidValue = -99999;
    static cv::Point s_drag_previous_point = { kInvalidValue, kInvalidValue };
    if (event == cv::EVENT_LBUTTONUP) {
        s_drag_previous_point.x = kInvalidValue;
        s_drag_previous_point.y = kInvalidValue;
    } else if (event == cv::EVENT_LBUTTONDOWN) {
        s_drag_previous_point.x = x;
        s_drag_previous_point.y = y;
    } else {
        if (s_drag_previous_point.x != kInvalidValue) {
            top_rvec[1] += kIncAnglePerPx * (x - s_drag_previous_point.x);
            top_rvec[0] -= kIncAnglePerPx * (y - s_drag_previous_point.y);
            s_drag_previous_point.x = x;
            s_drag_previous_point.y = y;
            image_processor->SetCameraParameter(f, real_rvec, real_tvec, top_rvec, top_tvec);
        }
    }
}


static void TreatKeyInputMain(std::unique_ptr<ImageProcessorIf>& image_processor, int32_t key, cv::VideoCapture& cap)
{
    float f;
    std::array<float, 3> real_rvec;
    std::array<float, 3> real_tvec;
    std::array<float, 3> top_rvec;
    std::array<float, 3> top_tvec;
    image_processor->GetCameraParameter(f, real_rvec, real_tvec, top_rvec, top_tvec);

    bool is_key_pressed = true;
    static constexpr float kIncPosPerFrame = 0.8f;
    key &= 0xFF;
    switch (key) {
    case 'w':
        top_tvec[2] -= kIncPosPerFrame;
        break;
    case 'W':
        top_tvec[2] -= kIncPosPerFrame * 3;
        break;
    case 's':
        top_tvec[2] += kIncPosPerFrame;
        break;
    case 'S':
        top_tvec[2] += kIncPosPerFrame * 3;
        break;
    case 'a':
        top_tvec[0] += kIncPosPerFrame;
        break;
    case 'A':
        top_tvec[0] += kIncPosPerFrame * 3;
        break;
    case 'd':
        top_tvec[0] -= kIncPosPerFrame;
        break;
    case 'D':
        top_tvec[0] -= kIncPosPerFrame * 3;
        break;
    case 'z':
        top_tvec[1] += kIncPosPerFrame;
        break;
    case 'Z':
        top_tvec[1] += kIncPosPerFrame * 3;
        break;
    case 'x':
        top_tvec[1] -= kIncPosPerFrame;
        break;
    case 'X':
        top_tvec[1] -= kIncPosPerFrame * 3;
        break;
    case 'q':
        top_rvec[2] += 0.1f;
        break;
    case 'e':
        top_rvec[2] -= 0.1f;
        break;
    default:
        is_key_pressed = false;
        switch (key) {
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
        break;
    }

    if (is_key_pressed) {
        image_processor->SetCameraParameter(f, real_rvec, real_tvec, top_rvec, top_tvec);
    }
}



int main(int argc, char* argv[])
{
    /*** Initialize ***/
    /* Find source image */
    std::string input_name = (argc > 1) ? argv[1] : DEFAULT_INPUT_IMAGE;
    cv::VideoCapture cap;   /* if cap is not opened, src is still image */
    if (!CommonHelper::FindSourceImage(input_name, cap)) {
        return -1;
    }

    /* Create video writer to save output video */
    cv::VideoWriter writer;
    // writer = cv::VideoWriter("out.mp4", cv::VideoWriter::fourcc('M', 'P', '4', 'V'), (std::max)(10.0, cap.get(cv::CAP_PROP_FPS)), cv::Size(static_cast<int32_t>(cap.get(cv::CAP_PROP_FRAME_WIDTH)), static_cast<int32_t>(cap.get(cv::CAP_PROP_FRAME_HEIGHT))));

    /* Initialize image processor library */
    std::unique_ptr<ImageProcessorIf> image_processor = ImageProcessorIf::Create();
    ImageProcessorIf::InputParam input_param = { WORK_DIR, 4 };
    if (image_processor->Initialize(input_param) != 0) {
        printf("Initialization Error\n");
        return -1;
    }

    /* Initialize cvui */
    cvui::init(kWindowNormal);
    cvui::init(kWindowTopView);
    cvui::init(kWindowParam);
    cv::setMouseCallback(kWindowTopView, CallbackMouseMain, image_processor.get());

    /*** Process for each frame ***/
    int32_t frame_cnt = 0;
    for (frame_cnt = 0; cap.isOpened() || frame_cnt < LOOP_NUM_FOR_TIME_MEASUREMENT; frame_cnt++) {
        if (loop_main(image_processor, frame_cnt, input_name, cap, writer) < 0) break;
        loop_param(image_processor);
        
        int32_t key = cv::waitKey(1);
        if (key == 27) break;   /* ESC to quit */
        TreatKeyInputMain(image_processor, key, cap);
    }


    /*** Finalize ***/
    /* Print average processing time */
    if (frame_cnt > 1) {
        frame_cnt--;    /* because the first process was not counted */
        printf("=== Average processing time ===\n");
        printf("Total:               %9.3lf [msec]\n", total_time_all / frame_cnt);
        printf("  Capture:           %9.3lf [msec]\n", total_time_cap / frame_cnt);
        printf("  Image processing:  %9.3lf [msec]\n", total_time_image_process / frame_cnt);
    }

    /* Fianlize image processor library */
    image_processor->Finalize();
    if (writer.isOpened()) writer.release();
    cv::waitKey(-1);

    return 0;
}

