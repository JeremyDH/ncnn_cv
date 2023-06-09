//
// Created by dWX1185603 on 2023/6/9.
//

#ifndef _MOBILENET_H_
#define _MOBILENET_H_
#include "../Classifier_Detecter.h"
#include "ncnn/net.h"
#include "string"

namespace mirror {
    class mobilenet : public Classifier_Detecter{
    public:
        mobilenet();
        ~mobilenet();
        int Loadmodel(const char* root_path);
        int Class_Detect(cv::Mat& img_src, std::vector<ImageInfo>* classifiers);

    private:
        ncnn::Net* mobileNet_;
        bool initialized_ ;
        float meanVal[3] = {103.94f, 116.78f, 123.68f };
        float normVal[3] = {0.017f,  0.017f,  0.017f };
        std::vector<std::string> labels_;
        const cv::Size inputSize = cv::Size(224, 224);

        int LoadLabels(const char* root_path);

    };
}

#endif // _MOBILENET_H_
