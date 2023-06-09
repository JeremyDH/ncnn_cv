//
// Created by dWX1185603 on 2023/6/8.
//

#ifndef _OBJECT_ENGINE_H
#define _OBJECT_ENGINE_H
#include "opencv2/opencv.hpp"
#include "vector"
#include "../common/common.h"

#if defined(_MSC_VER) || defined(_WIN32) || defined(_WIN64)
#ifdef OBJECT_EXPORTS
#define OBJECT_API __declspec(dllexport)
#else
#define OBJECT_API __declspec(dllimport)
#endif
#else
#define OBJECT_API __attribute__((visibility("default")))
#endif

namespace mirror {
    class Object_engine {
    public:
        OBJECT_API  Object_engine();
        OBJECT_API  ~Object_engine();
        OBJECT_API  int Loadmodel(const char* root_path);
        OBJECT_API  int DetectObject(const cv::Mat& img_src, std::vector<ObjectInfo>* objects);

    private:
        class Impl;
        Impl* impl_;
    };

}

#endif //_OBJECT_ENGINE_H
