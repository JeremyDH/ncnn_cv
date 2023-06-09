//
// Created by dWX1185603 on 2023/6/8.
//

#ifndef VISIONENGINE_MOBILENETSSD_H
#define VISIONENGINE_MOBILENETSSD_H

#include "../Object_detecter.h"
#include "ncnn/net.h"

namespace mirror {

    class mobilenetssd : public object_detecter{
    public:
        mobilenetssd();
        ~mobilenetssd();
        int Loadmodel(const char* root_path);
        int Object_d(const cv::Mat& img_src, std::vector<ObjectInfo>* objects);

    private:
        bool initialized_ ;
        ncnn::Net* mobilenetssd_ = nullptr;
        const float meanVals[3] = {0.5f, 0.5f, 0.5f};
        const float normVals[3] = {0.007843f, 0.007843f, 0.007843f};
        const float scoreThreshold_ = 0.7f;
        const float numsThredhold_ = 0.5f;
        std::vector<std::string> class_names = {
                "background", "aeroplane", "bicycle", "bird", "boat",
                "bottle", "bus", "car", "cat", "chair",
                "cow", "diningtable", "dog", "horse",
                "motorbike", "person", "pottedplant",
                "sheep", "sofa", "train", "tvmonitor"
        };
    };
}


#endif //_MOBILENETSSD_H
