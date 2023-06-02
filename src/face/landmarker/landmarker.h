//
// Created by dWX1185603 on 2023/6/2.
//

#ifndef _FACE_LANDMARKER_H_
#define _FACE_LANDMARKER_H_

#include "opencv2/imgproc.hpp"

namespace mirror {

    //抽象类
    class Landmarker {
    public:
        Landmarker();
        ~Landmarker();
        virtual int LoadModel(const char* root_path) = 0;
        virtual int ExtractKeypoints(const cv::Mat& img_src,
                                     const cv::Rect& face, std::vector<cv::Point2f>* keypoints) = 0;

    };

    //工厂基类
    class LandmarkerFactory{
    public:
        Landmarker* CreateLandmaker();
        ~LandmarkerFactory();
    };

    //不同landmark检测器工厂
    class InsightfaceLandmarkerFactory : public LandmarkerFactory{
    public:
        InsightfaceLandmarkerFactory(){}
        Landmarker* CreateLandmarker();
        ~InsightfaceLandmarkerFactory(){}
    };
}

#endif //_FACE_LANDMARKER_H_
