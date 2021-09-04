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
#ifndef IMAGE_PROCESSOR_IF_H_
#define IMAGE_PROCESSOR_IF_H_

/* for general */
#include <cstdint>
#include <memory>

namespace cv {
    class Mat;
};

class ImageProcessorIf {
public:
    typedef struct {
        char     work_dir[256];
        int32_t  num_threads;
    } InputParam;

    typedef struct {
        double time_pre_process;   // [msec]
        double time_inference;    // [msec]
        double time_post_process;  // [msec]
    } Result;

public:
    static std::unique_ptr<ImageProcessorIf> Create();

public:
    virtual ~ImageProcessorIf() {}
    virtual int32_t Initialize(const InputParam& input_param) = 0;
    virtual int32_t Process(cv::Mat& mat, Result& result) = 0;
    virtual int32_t Finalize(void) = 0;
    virtual int32_t Command(int32_t cmd) = 0;
};

#endif
