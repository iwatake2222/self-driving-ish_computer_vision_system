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
#include <memory>

/* for My modules */
#include "image_processor.h"
#include "image_processor_if.h"


/*** Macro ***/
#define TAG "ImageProcessorIf"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)


/*** Global variable ***/


/*** Function ***/
std::unique_ptr<ImageProcessorIf> ImageProcessorIf::Create()
{
    std::unique_ptr<ImageProcessorIf> ret(new ImageProcessor());
    return ret;
}
