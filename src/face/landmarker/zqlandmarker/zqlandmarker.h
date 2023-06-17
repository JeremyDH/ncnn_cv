//
// Created by Jeremy Dong on 2023/6/17.
//

#ifndef  _ZQLANDMARKER_H
#define  _ZQLANDMARKER_H
#include "../landmarker.h"
#include "ncnn/net.h"

namespace mirror {
    class ZQLandmarker : public Landmarker{
    public:
        ZQLandmarker();
        ~ZQLandmarker();

        int LoadModel(const char* root_path);
        int ExtractKeypoints(const cv::Mat& img_src,
                                     const cv::Rect& face, std::vector<cv::Point2f>* keypoints);

    private:
        ncnn::Net* zq_landmark_net_;
        const float meanVal[3] = {127.5f, 127.5f, 127.5f};
        const float normVal[3] = {0.0078125f, 0.0078125f, 0.0078125f};
        bool initialized;
    };
}

#endif //VISIONENGINE_ZQLANDMARKER_H
