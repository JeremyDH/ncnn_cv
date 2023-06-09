//
// Created by dWX1185603 on 2023/6/9.
//

#include "mobilenet.h"

namespace mirror{

    mobilenet::mobilenet() {
        mobileNet_ = new ncnn::Net();
        initialized_ = false;
    }

    mobilenet::~mobilenet() {
        if(mobileNet_){
            delete mobileNet_;
            mobileNet_ = nullptr;
        }
    }

    int mobilenet::Loadmodel(const char *root_path) {
        std::cout << "start mobilenet load model" << std::endl;
        std::string mo_param = std::string(root_path) + "/mobilenet.param";
        std::string mo_bin = std::string(root_path) + "/mobilenet.bin";

        std::cout << mo_param << std::endl;
        std::cout << mo_bin << std::endl;

        if(mobileNet_->load_param(mo_param.c_str()) == -1 ||
        mobileNet_->load_model(mo_bin.c_str()) == -1 ||
                LoadLabels(root_path) !=0 ){
            std::cout << "load param is failed" << std::endl;
            return 10000;
        }
        initialized_ = true;

        std::cout << "end mobilenet model" << std::endl;
        return 0;
    }

    int mobilenet::Class_Detect(cv::Mat &img_src, std::vector<ImageInfo> *classifiers) {
        std::cout << "start classify " << std::endl;
        classifiers->clear();
        if(img_src.empty())
        {
            std::cout << "load img_src is failed" << std::endl;
            return 10000;
        }
        if(!initialized_)
        {
            std::cout << "net weight is not load" << std::endl;
            return 10000;
        }
        //处理图像
        int width = img_src.cols;
        int height = img_src.rows;

        cv::Mat img_cp = img_src.clone();
        ncnn::Mat img_in = ncnn::Mat::from_pixels_resize(img_src.data, ncnn::Mat::PIXEL_BGR2RGB,
                                                         img_src.cols, img_src.rows, inputSize.width, inputSize.height);
        img_in.substract_mean_normalize(meanVal, normVal);

        ncnn::Extractor ex = mobileNet_->create_extractor();
        ex.input("data", img_in);
        ncnn::Mat out;
        ex.extract("prob", out);

        std::vector<std::pair<float, int>> scores;
        for(int i=0; i< out.w; ++i)
        {
            scores.push_back(std::make_pair(out[i], i));
        }

        int topK=5;
        std::partial_sort(scores.begin(), scores.begin()+ 5, scores.end(),
                          std::greater<std::pair<float, int>>());

        for(int i=0; i< topK; ++i)
        {
            ImageInfo image_info;
            image_info.label_ = labels_[scores[i].second];
            image_info.score_ = scores[i].first;
            classifiers->push_back(image_info);
        }
        std::cout << "start classify " << std::endl;
        return 0;
    }

    int mobilenet::LoadLabels(const char *root_path) {
        std::string  label_files = std::string(root_path) + "/label.txt";
        FILE* fp = fopen(label_files.c_str(), "r");

        while(!feof(fp)){
            char str[1024];
            if(nullptr == fgets(str, 1024, fp)) continue;
            std::string str_s(str);

            if(str_s.length() > 0)
            {
                for(int j=0; j<str_s.length(); j++){
                    if(str_s[j] == ' '){
                        std::string strr = str_s.substr(j, str_s.length() - j -1);
                        labels_.push_back(strr);
                        j = str_s.length();
                    }
                }
            }
        }
        return 0;
    }
}