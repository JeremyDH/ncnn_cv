//
// Created by dWX1185603 on 2023/6/6.
//

#ifndef FACE_MTCNN_H_
#define FACE_MTCNN_H_
#include "../detecter.h"
#include <vector>
#include <ncnn/net.h>

namespace mirror {
    class Mtcnnface : public Detecter{
    public:
        Mtcnnface();
        ~Mtcnnface();
        int LoadModel(const char* root_path);
        int DetectFace(const cv::Mat& img_src, std::vector<FaceInfo>* faces);

    private:
        ncnn::Net* P_Net = nullptr;
        ncnn::Net* R_Net = nullptr;
        ncnn::Net* O_Net = nullptr;
        bool initialized_;
        int pnet_size_;
        int min_face_size;
        float scale_factor_;
        const float meanVals[3] = { 127.5f, 127.5f, 127.5f };
        const float normVals[3] = { 0.0078125f, 0.0078125f, 0.0078125f };
        const float nms_threshold_[3] = { 0.5f, 0.7f, 0.7f };
        const float threshold_[3] = { 0.8f, 0.8f, 0.6f };

    private:
        int PDetect(const ncnn::Mat& img_in, std::vector<FaceInfo>* first_bboxes);
        int RDetect(const ncnn::Mat& img_in, const std::vector<FaceInfo>& first_bboxes,
                    std::vector<FaceInfo>* second_bboxes);
        int ODetect(const ncnn::Mat& img_in,
                    const std::vector<FaceInfo>& second_bboxes,
                    std::vector<FaceInfo>* third_bboxes);
        int Refine(std::vector<FaceInfo>* bboxes, const cv::Size max_size);

    };
}

#endif //VISIONENGINE_MTCNN_H
