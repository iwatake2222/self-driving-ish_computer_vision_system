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
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <iterator>

#include "common_helper.h"

/*** Macro ***/
#define TAG "CommonHelper"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

float CommonHelper::Sigmoid(float x)
{
    if (x >= 0) {
        return 1.0f / (1.0f + std::exp(-x));
    } else {
        return std::exp(x) / (1.0f + std::exp(x));    /* to aovid overflow */
    }
}

float CommonHelper::Logit(float x)
{
    if (x == 0) {
        return static_cast<float>(INT32_MIN);
    } else  if (x == 1) {
        return static_cast<float>(INT32_MAX);
    } else {
        return std::log(x / (1.0f - x));
    }
}

template<typename T>
T& CommonHelper::GetValue(std::vector<T>& val_list, std::vector<int32_t> shape, std::vector<int32_t> pos)
{
    static std::vector<int32_t> stride(shape.size());
    for (size_t i = 0; i < stride.size(); i++) {
        stride[i] = std::accumulate(shape.begin() + i + 1, shape.end(), 1, std::multiplies<int32_t>());
    }

    int32_t index = 0;
    for (size_t i = 0; i < pos.size(); i++) {
        index += pos[i] * stride[i];
    }
    return val_list.at(index);

}
template float& CommonHelper::GetValue<float>(std::vector<float>& val_list, std::vector<int32_t> shape, std::vector<int32_t> pos);
template int32_t& CommonHelper::GetValue<int32_t>(std::vector<int32_t>& val_list, std::vector<int32_t> shape, std::vector<int32_t> pos);
template int8_t& CommonHelper::GetValue<int8_t>(std::vector<int8_t>& val_list, std::vector<int32_t> shape, std::vector<int32_t> pos);
template uint8_t& CommonHelper::GetValue<uint8_t>(std::vector<uint8_t>& val_list, std::vector<int32_t> shape, std::vector<int32_t> pos);

template<typename T>
void CommonHelper::PrintValue(std::vector<T>& val_list, std::vector<int32_t> shape)
{
    size_t element_num = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int32_t>());
    if (val_list.size() != element_num) {
        PRINT_E("[PrintValue] invalid data size. %zd, %zd\n", val_list.size(), element_num);
        return;
    }

    std::vector<int32_t> stride(shape.size() - 1);
    for (size_t i = 0; i < stride.size(); i++) {
        stride[i] = std::accumulate(shape.begin() + i + 1, shape.end(), 1, std::multiplies<int32_t>());
    }

    for (size_t i = 0; i < element_num; i++) {
        printf("%3.4f, ", (float)val_list.at(i));
        for (size_t j = 0; j < stride.size(); j++) {
            if ((i + 1) % stride[j] == 0) printf("\n");
        }
    }
}
template void CommonHelper::PrintValue<float>(std::vector<float>& val_list, std::vector<int32_t> shape);
template void CommonHelper::PrintValue<int32_t>(std::vector<int32_t>& val_list, std::vector<int32_t> shape);
template void CommonHelper::PrintValue<int8_t>(std::vector<int8_t>& val_list, std::vector<int32_t> shape);
template void CommonHelper::PrintValue<uint8_t>(std::vector<uint8_t>& val_list, std::vector<int32_t> shape);
