//
// Created by dWX1185603 on 2023/6/9.
//

#include "iostream"
#include "../src/classifier/Classifier_Engine.h"
#include "opencv2/opencv.hpp"

using namespace mirror;

int main()
{
    //输入检测
    std::string img_path = "D:\\coding_learning\\ncnn_example\\data\\images\\cat.jpg";
    cv::Mat image_in = cv::imread(img_path);

    const char* root_path = "D:\\coding_learning\\ncnn_example\\model";

    Classifier_Engine* classif = new Classifier_Engine();
    classif->LoadModel(root_path);
    std::vector<mirror::ImageInfo> images;
    classif->Classifier(image_in, &images);
    int topk = images.size();
    for(int i=0; i<topk; ++i)
    {
        cv::putText(image_in, images[i].label_, cv::Point(10, 10+30 * i),
                    0, 0.5, cv::Scalar(255, 100, 0), 2, 2);
    }
    cv::imshow("result", image_in);
    cv::waitKey(0);
    delete classif;
    classif = nullptr;
    return 0;
}