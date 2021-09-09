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
#include "inference_helper.h"
#include "lane_engine.h"

/*** Macro ***/
#define TAG "LaneEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#if defined(ENABLE_TENSORRT)
#define MODEL_TYPE_ONNX
#else
#define MODEL_TYPE_TFLITE
#endif

#ifdef MODEL_TYPE_TFLITE
#define MODEL_NAME  "ultra_fast_lane_detection_culane_288x800.tflite"
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define INPUT_NAME  "input_1"
#define INPUT_DIMS  { 1, 288, 800, 3}
#define IS_NCHW     false
#define IS_RGB      true
#define OUTPUT_NAME "Identity"
#elif defined(MODEL_TYPE_ONNX)
#include "inference_helper_tensorrt.h"      // to call SetDlaCore
#define MODEL_NAME  "ultra_fast_lane_detection_culane_288x800.onnx"
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define INPUT_NAME  "input.1"
#define INPUT_DIMS  { 1, 3, 288, 800}
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME "200"
#endif

#if 0
static constexpr int32_t culane_row_anchor[] = { 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268, 272, 276, 280, 284 };
static constexpr int32_t kNumGriding = 101;
static constexpr int32_t kNumClassPerLine = 56;
static constexpr int32_t kNumLine = 4;
#else
static constexpr int32_t culane_row_anchor[] = { 121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287 };
static constexpr int32_t kNumGriding = 201;
static constexpr int32_t kNumClassPerLine = 18;
static constexpr int32_t kNumLine = 4;
#endif

static constexpr int32_t kNumWidth = 800;
static constexpr int32_t kNumHeight = 288;
static constexpr float kDeltaWidth = ((kNumWidth - 1) - 0) / static_cast<float>((kNumGriding - 1) - 1);

/*** Function ***/
int32_t LaneEngine::Initialize(const std::string& work_dir, const int32_t num_threads)
{
    /* Set model information */
    std::string model_filename = work_dir + "/model/" + MODEL_NAME;

    /* Set input tensor info */
    input_tensor_info_list_.clear();
    InputTensorInfo input_tensor_info(INPUT_NAME, TENSORTYPE, IS_NCHW);
    input_tensor_info.tensor_dims = INPUT_DIMS;
    input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
    /* 0 - 1.0 */
    input_tensor_info.normalize.mean[0] = 0;
    input_tensor_info.normalize.mean[1] = 0;
    input_tensor_info.normalize.mean[2] = 0;
    input_tensor_info.normalize.norm[0] = 1.0f;
    input_tensor_info.normalize.norm[1] = 1.0f;
    input_tensor_info.normalize.norm[2] = 1.0f;
    //input_tensor_info.normalize.mean[0] = 0.485f;
    //input_tensor_info.normalize.mean[1] = 0.456f;
    //input_tensor_info.normalize.mean[2] = 0.406f;
    //input_tensor_info.normalize.norm[0] = 0.229f;
    //input_tensor_info.normalize.norm[1] = 0.224f;
    //input_tensor_info.normalize.norm[2] = 0.225f;
    input_tensor_info_list_.push_back(input_tensor_info);

    /* Set output tensor info */
    output_tensor_info_list_.clear();
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME, TENSORTYPE));

    /* Create and Initialize Inference Helper */
#if defined(MODEL_TYPE_TFLITE)
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLite));
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteXnnpack));
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteGpu));
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteEdgetpu));
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteNnapi));
#elif defined(MODEL_TYPE_ONNX)
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorrt));
    // InferenceHelperTensorRt* p = dynamic_cast<InferenceHelperTensorRt*>(inference_helper_.get());
    // if (p) p->SetDlaCore(0);  /* Use DLA */
#endif

    if (!inference_helper_) {
        return kRetErr;
    }
    if (inference_helper_->SetNumThreads(num_threads) != InferenceHelper::kRetOk) {
        inference_helper_.reset();
        return kRetErr;
    }
    if (inference_helper_->Initialize(model_filename, input_tensor_info_list_, output_tensor_info_list_) != InferenceHelper::kRetOk) {
        inference_helper_.reset();
        return kRetErr;
    }

    return kRetOk;
}

int32_t LaneEngine::Finalize()
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    inference_helper_->Finalize();
    return kRetOk;
}


/* out_j = out_j[:, ::-1, :] */
static inline void Flip_1(std::vector<float>& val_list, int32_t num_i, int32_t num_j, int32_t num_k)
{
    for (int32_t i = 0; i < num_i; i++) {
        for (int32_t j = 0; j < num_j / 2; j++) {
            for (int32_t k = 0; k < num_k; k++) {
                int32_t new_j = num_j - 1 - j;
                std::swap(val_list.at(i * (num_j * num_k) + j * num_k + k), val_list.at(i * (num_j * num_k) + new_j * num_k + k));
            }
        }
    }
}

/* prob = scipy.special.softmax(out_j[:-1, :, :], axis=0) */
static std::vector<float> Softmax_0(const std::vector<float>& val_list, int32_t num_i, int32_t num_j, int32_t num_k)
{
    std::vector<float> res(val_list.size());
    for (int32_t j = 0; j < num_j; j++) {
        for (int32_t k = 0; k < num_k; k++) {
            float sum = 0;
            for (int32_t i = 0; i < num_i; i++) {
                sum += std::exp(val_list.at(i * (num_j * num_k) + j * num_k + k));
            }
            for (int32_t i = 0; i < num_i; i++) {
                float v =  std::exp(val_list.at(i * (num_j * num_k) + j * num_k + k)) / sum;
                res.at(i * (num_j * num_k) + j * num_k + k) = v;
            }
        }
    }
    return res;
}


/* loc = np.sum(prob * idx, axis=0) */
static std::vector<float> MulSum(const std::vector<float>& val_list_0, const std::vector<float>& val_list_1, int32_t num_i, int32_t num_j, int32_t num_k)
{
    std::vector<float> res(num_j * num_k);
    for (int32_t j = 0; j < num_j; j++) {
        for (int32_t k = 0; k < num_k; k++) {
            float sum = 0;
            for (int32_t i = 0; i < num_i; i++) {
                sum += val_list_0.at(i * (num_j * num_k) + j * num_k + k) * val_list_1.at(i);
            }
            res.at(j * num_k + k) = sum;
        }
    }
    return res;
}

/* out_j = np.argmax(out_j, axis = 0) */
/* loc[out_j == cfg.griding_num] = 0 */
static inline std::vector<bool> CheckIfValid(const std::vector<float>& val_list, int32_t num_i, int32_t num_j, int32_t num_k)
{
    std::vector<bool> res(num_j * num_k, true);
    for (int32_t j = 0; j < num_j; j++) {
        for (int32_t k = 0; k < num_k; k++) {
            float max_val = -999;
            int32_t max_index = 0;
            for (int32_t i = 0; i < num_i; i++) {
                float val = val_list.at(i * (num_j * num_k) + j * num_k + k);
                if (val > max_val) {
                    max_val = val;
                    max_index = i;
                }
            }
            if (max_index == num_i - 1) {
                res.at(j * num_k + k) = false;
            }
        }
    }
    return res;
}

int32_t LaneEngine::Process(const cv::Mat& original_mat, Result& result)
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    /*** PreProcess ***/
    const auto& t_pre_process0 = std::chrono::steady_clock::now();
    InputTensorInfo& input_tensor_info = input_tensor_info_list_[0];

    /* do resize and color conversion here because some inference engine doesn't support these operations */
    int32_t crop_x = 0;
    int32_t crop_y = 0;
    int32_t crop_w = original_mat.cols;
    int32_t crop_h = original_mat.rows;
    //int32_t crop_x = 0;
    //int32_t crop_w = original_mat.cols;
    //int32_t crop_h = (crop_w * kNumHeight) / kNumWidth;
    //int32_t crop_y = (original_mat.rows - crop_h) / 1;
    cv::Mat img_src = cv::Mat::zeros(input_tensor_info.GetHeight(), input_tensor_info.GetWidth(), CV_8UC3);
    CommonHelper::CropResizeCvt(original_mat, img_src, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeStretch);
    //CommonHelper::CropResizeCvt(original_mat, img_src, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeCut);
    //CommonHelper::CropResizeCvt(original_mat, img_src, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeExpand);

    input_tensor_info.data = img_src.data;
    input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
    input_tensor_info.image_info.width = img_src.cols;
    input_tensor_info.image_info.height = img_src.rows;
    input_tensor_info.image_info.channel = img_src.channels();
    input_tensor_info.image_info.crop_x = 0;
    input_tensor_info.image_info.crop_y = 0;
    input_tensor_info.image_info.crop_width = img_src.cols;
    input_tensor_info.image_info.crop_height = img_src.rows;
    input_tensor_info.image_info.is_bgr = false;
    input_tensor_info.image_info.swap_color = false;

    if (inference_helper_->PreProcess(input_tensor_info_list_) != InferenceHelper::kRetOk) {
        return kRetErr;
    }
    const auto& t_pre_process1 = std::chrono::steady_clock::now();

    /*** Inference ***/
    const auto& t_inference0 = std::chrono::steady_clock::now();
    if (inference_helper_->Process(output_tensor_info_list_) != InferenceHelper::kRetOk) {
        return kRetErr;
    }
    const auto& t_inference1 = std::chrono::steady_clock::now();

    /*** PostProcess ***/
    const auto& t_post_process0 = std::chrono::steady_clock::now();

    /* Retrieve the result */
    std::vector<float> output_raw_val(output_tensor_info_list_[0].GetDataAsFloat(), output_tensor_info_list_[0].GetDataAsFloat() + kNumGriding * kNumClassPerLine * kNumLine);
    if (output_tensor_info_list_[0].GetElementNum() != kNumGriding * kNumClassPerLine * kNumLine) {
        PRINT_E("Invalid output\n");
        return kRetErr;
    }

    /* reference: https://github.com/cfzd/Ultra-Fast-Lane-Detection/blob/master/demo.py#L69 */
    std::vector<float> prob = Softmax_0(output_raw_val, kNumGriding - 1, kNumClassPerLine, kNumLine);
    std::vector<float> idx(kNumGriding - 1);
    std::iota(idx.begin(), idx.end(), 1.0f);
    std::vector<float> loc = MulSum(prob, idx, kNumGriding - 1, kNumClassPerLine, kNumLine);
    std::vector<bool> valid_map = CheckIfValid(output_raw_val, kNumGriding, kNumClassPerLine, kNumLine);

    for (int32_t k = 0; k < kNumLine; k++) {
        Line line;
        for (int32_t j = 0; j < kNumClassPerLine; j++) {
            int32_t index = j * kNumLine + k;
            float val = loc.at(index);
            if (valid_map.at(index) && val > 0) {
                int32_t x = static_cast<int32_t>(val * kDeltaWidth * crop_w / kNumWidth + crop_x);
                int32_t y = static_cast<int32_t>(culane_row_anchor[j] * crop_h / kNumHeight + crop_y);
                line.push_back({ x, y });
            }
        }
        result.line_list.push_back(line);
    }
    const auto& t_post_process1 = std::chrono::steady_clock::now();


    /* Return the results */
    result.crop.x = (std::max)(0, crop_x);
    result.crop.y = (std::max)(0, crop_y);
    result.crop.w = (std::min)(crop_w, original_mat.cols - result.crop.x);
    result.crop.h = (std::min)(crop_h, original_mat.rows - result.crop.y);
    result.time_pre_process = static_cast<std::chrono::duration<double>>(t_pre_process1 - t_pre_process0).count() * 1000.0;
    result.time_inference = static_cast<std::chrono::duration<double>>(t_inference1 - t_inference0).count() * 1000.0;
    result.time_post_process = static_cast<std::chrono::duration<double>>(t_post_process1 - t_post_process0).count() * 1000.0;;

    return kRetOk;
}
