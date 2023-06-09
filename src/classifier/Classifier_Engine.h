//
// Created by dWX1185603 on 2023/6/9.
//

#ifndef _CLASSIFIER_ENGINE_H
#define _CLASSIFIER_ENGINE_H
#include <opencv2/opencv.hpp>
#include <vector>
#include "../common/common.h"

namespace mirror {
    class Classifier_Engine {
    public:
        Classifier_Engine();
        ~Classifier_Engine();
        int LoadModel(const char* root_path);
        int Classifier(cv::Mat& img_src, std::vector<ImageInfo>* classifers);

    private:
        class Impl_;
        Impl_* impl_;

    };

}

#endif //_CLASSIFIER_ENGINE_H
