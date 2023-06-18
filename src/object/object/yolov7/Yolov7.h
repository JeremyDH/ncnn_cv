//
// Created by Jeremy Dong on 2023/6/17.
//

#ifndef VISIONENGINE_YOLOV7_H
#define VISIONENGINE_YOLOV7_H

#include "../Object_detecter.h"
#include "ncnn/net.h"

#define MAX_STRIDE 32

namespace mirror {
    class Yolov7 : public object_detecter{
        Yolov7();
        ~Yolov7();
        int LoadModel(const char* root_path);
        int Object_d(const cv::Mat& img_src, std::vector<ObjectInfo>* objects);

    };
}

#endif //VISIONENGINE_YOLOV7_H
