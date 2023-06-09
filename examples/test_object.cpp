//
// Created by dWX1185603 on 2023/6/8.
//
#define OBJECT_EXPORTS
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <string>
#include "../src/object/object_engine.h"

using namespace mirror;

int objects_demo() {
        std::string img_path = "./data/images/dog.jpg";
        const char* root_path = "model";
        cv::Mat img_src = cv::imread(img_path);
        //开始时间
        double start = static_cast<double>(cv::getTickCount());

        Object_engine *ob_engine = new Object_engine();
        ob_engine->Loadmodel(root_path);
        std::vector<ObjectInfo> outs;

        std::cout << "loadding model is  start" << std::endl;
        ob_engine->DetectObject(img_src, &outs);

        int nums = static_cast<int>(outs.size());
        for(int i=0; i< nums; ++i)
        {
            cv::rectangle(img_src, outs[i].location_, cv::Scalar(255, 0, 255), 2);

            char text[256];
            sprintf(text, "%s %.1f%%", outs[i].name_.c_str(), outs[i].score_ * 100);
            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            cv::putText(img_src, text, cv::Point(outs[i].location_.x, outs[i].location_.y+label_size.height),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
        }
        double end = static_cast<double>(cv::getTickCount());
        double time_cost = (end - start) / cv::getTickFrequency() * 1000;

        cv::imwrite("data/images/ob.jpg", img_src);
        cv::imshow("reslut", img_src);
        std::cout << "using time: " << time_cost << std::endl;
        cv::waitKey(0);

        delete ob_engine;
        ob_engine = nullptr;


        return 0;
}

int main()
{
    objects_demo();
}