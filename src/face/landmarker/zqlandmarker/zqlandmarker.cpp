//
// Created by Jeremy Dong on 2023/6/17.
//

#include "zqlandmarker.h"
#include <iostream>
#include <string>



namespace mirror
{
    ZQLandmarker::ZQLandmarker() {
        zq_landmark_net_ = new ncnn::Net();
        initialized = false;
    }
    ZQLandmarker::~ZQLandmarker() noexcept {
        if(zq_landmark_net_)
        {
            delete zq_landmark_net_;
            zq_landmark_net_ = nullptr;
        }
    }
    int ZQLandmarker::LoadModel(const char *root_path) {

        std::cout << "Load ZQLandmarker model" << std::endl;
        std::string zq_param = std::string(root_path) + "/fl.param";
        std::string zq_bin = std::string(root_path) + "/fl.bin";

        if(zq_landmark_net_->load_param(zq_param.c_str()) == -1 ||
           zq_landmark_net_->load_model(zq_bin.c_str()) == -1)
        {
            std::cout << "Load zq_landmark_net_ model is failed" << std::endl;
            return 1000;
        }
        initialized = true;
        std::cout << "load zq_landmark_net_ model success" << std::endl;
        return 0;
    }

    int ZQLandmarker::ExtractKeypoints(const cv::Mat &img_src, const cv::Rect &face,
                                       std::vector<cv::Point2f> *keypoints) {
        keypoints->clear();
        std::cout << "ZQLandmarker start extractKeypoints" << std::endl;
        if(!initialized)
        {
            std::cout << "zq landmarker unitialized." << std::endl;
            return 10000;
        }
        if(img_src.empty())
        {
            std::cout << "input empty" << std::endl;
            return 10001;
        }

        cv::Mat img_face = img_src(face).clone();
        ncnn::Mat img_in = ncnn::Mat::from_pixels_resize(img_face.data, ncnn::Mat::PIXEL_BGR2RGB,
                                                         img_face.cols, img_face.rows, 112, 112);
        img_in.substract_mean_normalize(meanVal, normVal);
        ncnn::Extractor ex = zq_landmark_net_->create_extractor();
        ex.input("data", img_in);
        ncnn::Mat out;
        ex.extract("bn6_3", out);

        for(int i = 0; i < 106; i++)
        {
            float x = abs(out[2 * i] * img_face.cols) + face.x;
            float y = abs(out[2 * i + 1] * img_face.rows) + face.y;
            keypoints->push_back(cv::Point2f(x, y));
        }

        std::cout << "end extract keypoints" << std::endl;
        return 0;

    }
}
