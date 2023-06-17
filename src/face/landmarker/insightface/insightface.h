//
// Created by dWX1185603 on 2023/6/2.
//

#ifndef _FACE_INSIGHTFACE_H_
#define _FACE_INSIGHTFACE_H_

#include "../landmarker.h"
#include "ncnn/net.h"

namespace mirror {
    class Insightface : public Landmarker{
    public:
        Insightface();
        ~Insightface();

        int LoadModel(const char* root_path);
        int ExtractKeypoints(const cv::Mat& img_src,
                             const cv::Rect& face, std::vector<cv::Point2f>* keypoints);

    private:
        ncnn::Net* insightface_landmarker_net_;
        bool initialized_;

    };
}

#endif //_FACE_INSIGHTFACE_H_
