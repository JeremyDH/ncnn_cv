//
// Created by dWX1185603 on 2023/6/8.
//

#include "mobilenetssd.h"

#include <iostream>
#include <vector>


namespace mirror{
    mobilenetssd::mobilenetssd() :
        mobilenetssd_(new ncnn::Net()),
        initialized_(false){}

    mobilenetssd::~mobilenetssd() noexcept {
        mobilenetssd_->clear();
    }

    int mobilenetssd::Loadmodel(const char *root_path) {
        std::cout << "start mobilenetssd model" << std::endl;

        std::string ssd_param = std::string(root_path) + "/mobilenetssd.param";
        std::string ssd_bin = std::string(root_path) + "/mobilenetssd.bin";

        if(mobilenetssd_->load_param(ssd_param.c_str()) == -1 ||
           mobilenetssd_->load_model(ssd_bin.c_str()) == -1){
            std::cout << "load mobilenetssd model failed" << std::endl;
            return 10000;
        }

        initialized_ = true;
        return 0;
    }

    int mobilenetssd::Object_d(const cv::Mat &img_src, std::vector<ObjectInfo> *objects)
    {
        std::cout << "start Object detect" << std::endl;

        objects->clear();
        if(img_src.empty()){
            std::cout << "input img_src is failed" << std::endl;
            return 10000;
        }
        if(!initialized_){
            std::cout << "initialized is fail" << std::endl;
            return 10000;
        }
        int width = img_src.cols;
        int height = img_src.rows;

        ncnn::Mat img_in = ncnn::Mat::from_pixels_resize(img_src.data, ncnn::Mat::PIXEL_BGR2RGB,
                                                  img_src.cols, img_src.rows, 300, 300);

        img_in.substract_mean_normalize(meanVals, normVals);

        ncnn::Extractor ex = mobilenetssd_->create_extractor();
//        ex.set_light_mode(true);
//        ex.set_num_threads(2);
        ex.input("data", img_in);
        ncnn::Mat out;
        ex.extract("detection_out", out);
        std::vector<ObjectInfo> obejects_tmp;

        for(int i=0; i<out.h; ++i)
        {
            const float* values = out.row(i);
            ObjectInfo object;
            object.name_ = class_names[int(values[0])];
            object.score_ = values[1];
            object.location_.x = values[2] * width;
            object.location_.y = values[3] * height;
            object.location_.width = values[4] * width - object.location_.x;
            object.location_.height = values[5] * height - object.location_.y;

            //filte the result
            if(object.score_ < scoreThreshold_){
                continue;
            }
            obejects_tmp.push_back(object);
        }
        NMS(obejects_tmp, objects, numsThredhold_);
        std::cout << "object number: " << objects->size() << std::endl;
        std::cout << "end object detect" << std::endl;

        return 0;
    }
}