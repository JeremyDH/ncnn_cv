//
// Created by dWX1185603 on 2023/6/9.
//

#ifndef _CLASSIFIER_DETECTER_H_
#define _CLASSIFIER_DETECTER_H_
#include "opencv2/opencv.hpp"
#include "vector"
#include "../Classifier_Engine.h"

namespace mirror {
    class Classifier_Detecter {
    public:
        virtual ~Classifier_Detecter(){}
        virtual int Loadmodel(const char* root_path) = 0;
        virtual int Class_Detect(cv::Mat& img_src, std::vector<ImageInfo>* classifiers) = 0;
    };

    class Classifier_Factory{
    public:
        virtual ~Classifier_Factory(){}
        virtual Classifier_Detecter* createClassifier() = 0;
    };

    class Mobilenet_Factory : public Classifier_Factory{
    public:
        Mobilenet_Factory(){}
        ~Mobilenet_Factory(){}
        Classifier_Detecter* createClassifier();
    };
}


#endif //_CLASSIFIER_DETECTER_H_
