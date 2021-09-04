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

/* for My modules */
#include "image_processor_if.h"
#include "common_helper_cv.h"

/*** Macro ***/
#define WORK_DIR                      RESOURCE_DIR
#define DEFAULT_INPUT_IMAGE           RESOURCE_DIR"/dashcam_00.jpg"
#define LOOP_NUM_FOR_TIME_MEASUREMENT 10

/*** Function ***/
int32_t main(int argc, char* argv[])
{
    /*** Initialize ***/
    /* variables for processing time measurement */
    double total_time_all = 0;
    double total_time_cap = 0;
    double total_time_image_process = 0;

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

    /*** Process for each frame ***/
    int32_t frame_cnt = 0;
    for (frame_cnt = 0; cap.isOpened() || frame_cnt < LOOP_NUM_FOR_TIME_MEASUREMENT; frame_cnt++) {
        const auto& time_all0 = std::chrono::steady_clock::now();
        /* Read image */
        const auto& time_cap0 = std::chrono::steady_clock::now();
        cv::Mat image;
        if (cap.isOpened()) {
            cap.read(image);
        } else {
            image = cv::imread(input_name);
        }
        if (image.empty()) break;
        const auto& time_cap1 = std::chrono::steady_clock::now();

        /* Call image processor library */
        const auto& time_image_process0 = std::chrono::steady_clock::now();
        ImageProcessorIf::Result result;
        image_processor->Process(image, result);
        const auto& time_image_process1 = std::chrono::steady_clock::now();

        /* Display result */
        if (writer.isOpened()) writer.write(result.mat_output);
        cv::imshow("output", result.mat_output);
        cv::imshow("output_topview", result.mat_output_topview);

        /* Input key command */
        if (cap.isOpened()) {
            /* this code needs to be before calculating processing time because cv::waitKey includes image output */
            /* however, when 'q' key is pressed (cap.released()), processing time significantly incraeases. So escape from the loop before calculating time */
            if (CommonHelper::InputKeyCommand(cap)) break;
        };

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
