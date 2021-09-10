# !!! Under Development !!!

# Self-Driving-ish Computer Vision System
- This project generates images you often see autonomous driving demo
- Detection
    - Object Detection and Tracking
    - Lane Detection and Curve Fitting
    - Road Segmentation
    - Depth Estimation
- Transform
    - Top View
    - Distance Calculation


# Tested Environment
## Computer
- Windows 10 (x64) + Visual Studio 2019
    - Intel Core i7-6700@3.4GHz + NVIDIA GeForce GTX 1070
- Jetson Xavier NX. JetPack 4.6

## Deep Learning Inference Framework
- TensorFlow Lite with XNNPACK delegate
    - CPU only
- TensorRT
    - GPU
- * Depth esstimation is supported with TensorRT only


# How to Build and Run
## Requirements
- OpenCV 4.x
- CMake

## Common 
- Get source code
    ```sh
    git clone https://github.com/iwatake2222/xxx.git
    cd xxx
    git submodule update --init --recursive --recommend-shallow --depth 1
    # You don't need the following lines if you use TensorRT
    cd inference_helper/third_party/tensorflow
    chmod +x tensorflow/lite/tools/make/download_dependencies.sh
    tensorflow/lite/tools/make/download_dependencies.sh
    ```
- Download prebuilt library (You don't need this step if you use TensorRT)
    - Download prebuilt libraries (third_party.zip) from https://github.com/iwatake2222/InferenceHelper/releases/  (<- Not in this repository)
    - Extract it to `inference_helper/third_party/`
- Download models
    - Download models (resource.zip) from https://github.com/iwatake2222/xxx/releases/ 
    - Extract it to `resource/`

## Windows (Visual Studio)
- Configure and Generate a new project using cmake-gui for Visual Studio 2019 64-bit
    - `Where is the source code` : path-to-cloned-folder
    - `Where to build the binaries` : path-to-build	(any)
- Open `xxx.sln`
- Set `xxx` project as a startup project, then build and run!
- Note:
    - Running with `Debug` causes exception, so use `Release` or `RelWithDebInfo` if you use TensorFlow Lite
    - You may need to modify cmake setting for TensorRT for your environment

## Linux (Jetson Xavier NX)
```sh
mkdir build && cd build
# cmake .. -DENABLE_TENSORRT=off
cmake .. -DENABLE_TENSORRT=on
make
./main
```

## cmake options
```sh
cmake .. -DENABLE_TENSORRT=off  # Use TensorFlow Lite (default)
cmake .. -DENABLE_TENSORRT=on   # Use TensorRT
```

## Usage
```
./main [input]
 - input = blank
    - use the default image file set in source code (main.cpp)
    - e.g. ./main
 - input = *.mp4, *.avi, *.webm
    - use video file
    - e.g. ./main test.mp4
 - input = *.jpg, *.png, *.bmp
    - use image file
    - e.g. ./main test.jpg
 - input = number (e.g. 0, 1, 2, ...)
    - use camera
    - e.g. ./main 0
- input = jetson
    - use camera via gstreamer on Jetson
    - e.g. ./main jetson
```


# Model Information
## Details
- Object Detection
    - YOLOX-Nano, 480x640
    - https://github.com/PINTO0309/PINTO_model_zoo/blob/main/132_YOLOX/download_nano_new.sh
    - https://github.com/PINTO0309/PINTO_model_zoo/blob/main/132_YOLOX/download_nano.sh
- Lane Detection
    - Ultra-Fast-Lane-Detection, 288x800
    - https://github.com/PINTO0309/PINTO_model_zoo/blob/main/140_Ultra-Fast-Lane-Detection/download_culane.sh
- Road Segmentation
    - road-segmentation-adas-0001, 512x896
    - https://github.com/PINTO0309/PINTO_model_zoo/blob/main/136_road-segmentation-adas-0001/download.sh
- Depth Estimation
    - LapDepth, 256x512
    - [00_doc/pytorch_pkl_2_onnx_LapDepth.ipynb](00_doc/pytorch_pkl_2_onnx_LapDepth.ipynb)

## Performance
| Model  | Jetson Xavier NX | GTX 1070 |
| ------ | ---------------: | -------: |
|  OD    |           9.2 ms |   9.2 ms |
|  Lane  |           7.6 ms |   7.6 ms |
|  Road  |          27.7 ms |  27.7 ms |
|  Depth |          54.7 ms |  54.7 ms |
|  Total |         10.0 fps | 10.0 fps |

*"Total" includes image capture, pre/post process, other image process, result image drawing, etc.


# License
- Copyright 2021 iwatake2222
- Licensed under the Apache License, Version 2.0
    - [LICENSE](LICENSE)


# Acknowledgements
## Code, Library
- TensorFlow
    - https://github.com/tensorflow/tensorflow
    - Copyright 2019 The TensorFlow Authors
    - Licensed under the Apache License, Version 2.0
    - Generated pre-built library from this project
- TensorRT
    - https://github.com/nvidia/TensorRT
    - Copyright 2020 NVIDIA Corporation
    - Licensed under the Apache License, Version 2.0
    - Cited source code
- cvui
    - https://github.com/Dovyski/cvui
    - Copyright (c) 2016 Fernando Bevilacqua
    - Licensed under the MIT License (MIT)
    - Cited source code
## Model
- PINTO_model_zoo
    - https://github.com/PINTO0309/PINTO_model_zoo
    - Copyright (c) 2019 Katsuya Hyodo
    - Licensed under the MIT License (MIT)
    - Cited converted model files
- YOLOX
    - https://github.com/Megvii-BaseDetection/YOLOX
    - Copyright (c) Megvii, Inc. and its affiliates. All Rights Reserved
    - Licensed under the Apache License, Version 2.0
    ```BibTeX
    @article{yolox2021,
        title={YOLOX: Exceeding YOLO Series in 2021},
        author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
        journal={arXiv preprint arXiv:2107.08430},
        year={2021}
    }
    ```
- Ultra-Fast-Lane-Detection
    - https://github.com/cfzd/Ultra-Fast-Lane-Detection
    - Copyright (c) 2020 cfzd
    - Licensed under the MIT License (MIT)
    ```BibTeX
    @InProceedings{qin2020ultra,
        author = {Qin, Zequn and Wang, Huanyu and Li, Xi},
        title = {Ultra Fast Structure-aware Deep Lane Detection},
        booktitle = {The European Conference on Computer Vision (ECCV)},
        year = {2020}
    }
    ```
- road-segmentation-adas-0001
    - https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/road-segmentation-adas-0001
    - Copyright (c) 2021 Intel Corporation
    - Licensed under the Apache License, Version 2.0
- LapDepth-release
    - https://github.com/tjqansthd/LapDepth-release
    - Licensed under the GNU General Public License v3.0
    ```BibTeX
    @ARTICLE{9316778,
        author={M. {Song} and S. {Lim} and W. {Kim}},
        journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
        title={Monocular Depth Estimation Using Laplacian Pyramid-Based Depth Residuals}, 
        year={2021},
        volume={},
        number={},
        pages={1-1},
        doi={10.1109/TCSVT.2021.3049869}}
    ```
## Image
- OpenCV
    - https://github.com/opencv/opencv
    - Licensed under the Apache License, Version 2.0
- Others
    - https://www.youtube.com/watch?v=tTuUjnISt9s
    - Licensed under the Creative Commons license
    - Copyright Dashcam Roadshow 2020
