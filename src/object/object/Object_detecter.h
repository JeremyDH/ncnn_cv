//
// Created by dWX1185603 on 2023/6/8.
//

#ifndef _OBJECT_DETECTER_H
#define _OBJECT_DETECTER_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "../../common/common.h"

namespace mirror{
    class object_detecter{
    public:
        virtual ~object_detecter(){}
        virtual int Loadmodel(const char* root_path) = 0;
        virtual int Object_d(const cv::Mat& img_src, std::vector<ObjectInfo>* objects) = 0;
    };

    class Obeject_Detect_Factory{
    public:
        virtual object_detecter* CreateMobilenetssd() = 0;
        virtual ~Obeject_Detect_Factory(){};
    };

  //各类目标检测器
    class MobilenetssdFactory : public Obeject_Detect_Factory{
    public:
        MobilenetssdFactory(){}
        ~MobilenetssdFactory(){}
        object_detecter* CreateMobilenetssd();
    };
}

#endif //_OBJECT_DETECTER_H
