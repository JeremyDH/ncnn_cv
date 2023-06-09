//
// Created by dWX1185603 on 2023/6/1.
//

#ifndef _FACE_CENTERFACE_H_
#define _FACE_CENTERFACE_H_
#include "../detecter.h"
#include "ncnn/net.h"
#include <vector>
#include "opencv2/opencv.hpp"

namespace mirror {
    class Centerface: public Detecter {
    public:
          Centerface();
          ~Centerface();
          int LoadModel(const char* root_path);
          int DetectFace(const cv::Mat& img_src, std::vector<FaceInfo>* faces);
    private:
        ncnn::Net* center_net_ = nullptr;
        bool initialized_;
        const float scoreThreshold_ = 0.8f;
        const float nmsThreshold_ = 0.6f;
    };

}
#endif //_FACE_CENTERFACE_H_
