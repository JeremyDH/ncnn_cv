//
// Created by dWX1185603 on 2023/6/6.
//
#include "../src/face/face_engine.h"
#include "opencv2/opencv.hpp"

using namespace mirror;

int TestLanfmark(int argc, char* argv[])
{
    const char* img_file = "./data/images/face2.jpg";
    cv::Mat img_src = cv::imread(img_file);
    const char* root_path = "./model";

    double start = static_cast<double>(cv::getTickCount());

    FaceEngine* face_engine = new FaceEngine();
    face_engine->LoadModel(root_path);
    std::vector<FaceInfo> faces;
    face_engine->DetectFace(img_src, &faces);
    for(int i=0; i<static_cast<int>(faces.size()); ++i)
    {
        cv::Rect face = faces.at(i).location_;
    }

}