//
// Created by dWX1185603 on 2023/6/2.
//

#include "insightface.h"
#include "iostream"
#include "string"
#include "../../../common/common.h"

namespace mirror
{
    Insightface::Insightface():
    insightface_landmarker_net_(new ncnn::Net()),
    initialized_(false){}

    Insightface::~Insightface() {
        if(insightface_landmarker_net_){
            insightface_landmarker_net_->clear();
        }
    }

    int Insightface::LoadModel(const char* root_path) {
        //加载模型
        std::string param_path = std::string(root_path) + "/2d106.param";
        std::string bin_path = std::string(root_path) + "/2d106.bin";

        if(insightface_landmarker_net_->load_param(param_path.c_str()) == -1 ||
        insightface_landmarker_net_->load_model(bin_path.c_str()) == -1){
            std::cout << "Load model fail" << std::endl;
            return -1;
        }
        initialized_ = true ;
    }

    int Insightface::ExtractKeypoints(const cv::Mat &img_src, const cv::Rect &face,
                                      std::vector<cv::Point2f> *keypoints) {
        keypoints->clear();
        if (!initialized_){
            std::cout << "Model is Uninitial" << std::endl;
            return 10000;
        }
        if (img_src.empty())
        {
            std::cout << "img is empty" << std::endl;
            return 10000;
        }
        //1 enlarge thr face rect
        cv::Rect face_enlarged = face;
        const float enlarge_scale = 1.5f;
        EnlargeRect(enlarge_scale, &face_enlarged);
        //2 square the rect
        RectifyRect(&face_enlarged);
        face_enlarged = face_enlarged & cv::Rect (0, 0, img_src.cols, img_src.rows);

        //3 crop the face
        cv::Mat img_face = img_src(face_enlarged).clone();
        //4 do inference
        ncnn::Extractor ex = insightface_landmarker_net_->create_extractor();
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_face.data,
                                                     ncnn::Mat::PIXEL_BGR2RGB, img_face.cols, img_face.rows, 192, 192);
        ex.input("data", in);
        ncnn::Mat out;
        ex.extract("fc1", out);
        for(int i=0; i<106; i++)
        {
            float x = (out[2 * i] + 1.0f) * img_face.cols / 2  + face_enlarged.x;
            float y = (out[2 * i + 1] + 1.0f) * img_face.rows / 2 + face_enlarged.y;
            keypoints->push_back(cv::Point2f(x, y));
        }
        return 0;

    }


}