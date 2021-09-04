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
/* for general */
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>

/* for My modules */
#include "bounding_box.h"


float BoundingBoxUtils::CalculateIoU(const BoundingBox& obj0, const BoundingBox& obj1)
{
    int32_t interx0 = (std::max)(obj0.x, obj1.x);
    int32_t intery0 = (std::max)(obj0.y, obj1.y);
    int32_t interx1 = (std::min)(obj0.x + obj0.w, obj1.x + obj1.w);
    int32_t intery1 = (std::min)(obj0.y + obj0.h, obj1.y + obj1.h);
    if (interx1 < interx0 || intery1 < intery0) return 0;

    int32_t area0 = obj0.w * obj0.h;
    int32_t area1 = obj1.w * obj1.h;
    int32_t areaInter = (interx1 - interx0) * (intery1 - intery0);
    int32_t areaSum = area0 + area1 - areaInter;

    return static_cast<float>(areaInter) / areaSum;
}



void BoundingBoxUtils::Nms(std::vector<BoundingBox>& bbox_list, std::vector<BoundingBox>& bbox_nms_list, float threshold_nms_iou, bool check_class_id)
{
    std::sort(bbox_list.begin(), bbox_list.end(), [](BoundingBox const& lhs, BoundingBox const& rhs) {
        //if (lhs.w * lhs.h > rhs.w * rhs.h) return true;
        if (lhs.score > rhs.score) return true;
        return false;
        });

    std::unique_ptr<bool[]> is_merged(new bool[bbox_list.size()]);
    for (size_t i = 0; i < bbox_list.size(); i++) is_merged[i] = false;
    for (size_t index_high_score = 0; index_high_score < bbox_list.size(); index_high_score++) {
        std::vector<BoundingBox> candidates;
        if (is_merged[index_high_score]) continue;
        candidates.push_back(bbox_list[index_high_score]);
        for (size_t index_low_score = index_high_score + 1; index_low_score < bbox_list.size(); index_low_score++) {
            if (is_merged[index_low_score]) continue;
            if (check_class_id && bbox_list[index_high_score].class_id != bbox_list[index_low_score].class_id) continue;
            if (CalculateIoU(bbox_list[index_high_score], bbox_list[index_low_score]) > threshold_nms_iou) {
                candidates.push_back(bbox_list[index_low_score]);
                is_merged[index_low_score] = true;
            }
        }

        bbox_nms_list.push_back(candidates[0]);
    }
}

void BoundingBoxUtils::FixInScreen(BoundingBox& bbox, int32_t width, int32_t height)
{
    bbox.x = (std::max)(0, bbox.x);
    bbox.y = (std::max)(0, bbox.y);
    bbox.w = (std::min)(width - bbox.x, bbox.w);
    bbox.h = (std::min)(width - bbox.y, bbox.h);
}