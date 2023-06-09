//
// Created by dWX1185603 on 2023/5/30.
//

#ifndef _DETECTER_H
#define _DETECTER_H

#include "opencv2/core.hpp"
#include "../../common/common.h"

namespace mirror
{
    //抽象类检测器
    class Detecter {
    public:
        virtual ~Detecter() {};
        virtual int LoadModel(const char* root_path) = 0;
        virtual int DetectFace(const cv::Mat& img_src, std::vector<FaceInfo>* faces) = 0;
    };

    //工厂基类
    class DetecterFactory
    {
    public:
        virtual Detecter* CreateDetecter() = 0;
        virtual ~DetecterFactory() {};
    };

    //不同人脸检测器
    class CenterfaceFactory : public DetecterFactory
    {
    public:
        CenterfaceFactory() {}
        ~CenterfaceFactory() {}
        Detecter* CreateDetecter();
    };
    class RetinafaceFactory :public DetecterFactory
    {
    public:
        RetinafaceFactory(){}
        ~RetinafaceFactory() {}
        Detecter* CreateDetecter();
    };

    //mtcnn人脸检测
    class MtcnnfaceFactory : public DetecterFactory
    {
    public:
        MtcnnfaceFactory(){}
        ~MtcnnfaceFactory(){}
        Detecter* CreateDetecter();
    };
}

#endif //_DETECTER_H
