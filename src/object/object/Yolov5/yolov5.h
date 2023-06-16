//
// Created by dWX1185603 on 2023/6/13.
//

#ifndef _YOLOV5_H
#define _YOLOV5_H
#include "ncnn/layer.h"
#include "ncnn/net.h"
#if defined(USE_NCNN_SIMPLEOCV)
#include "ncnn/simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include "../Object_detecter.h"
#include "float.h"
#include "stdio.h"
#include "vector"

#define YOLOV5_V62 1
#if YOLOV5_V60 || YOLOV5_V62
#define MAX_STRIDE 64
#else
#define MAX_STRIDE 32
class YoloV5Focus : public ncnn::Layer
{
public:
    YoloV5Focus()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int outw = w / 2;
        int outh = h / 2;
        int outc = channels * 4;

        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
        if(top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for(int p=0; p < outc; p++)
            {
            const float* ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) +((p / channels) / 2);
            float* outptr = top_blob.channel(p);

            for(int i = 0; i < outh; i++)
                {
                for(int j = 0; j < outw; j++)
                    {
                       *outptr = *ptr;

                       outptr += 1;
                       ptr += 2;

                    }
                    ptr += w;
                }
            }
            return 0;

    }
};

DEFINE_LAYER_CREATOR(YoloV5Focus)
#endif


namespace mirror {

    class Yolov5 : public object_detecter{
    public:
         Yolov5();
         ~Yolov5();
         int Loadmodel(const char* root_path);
         int Object_d(const cv::Mat& img_src, std::vector<ObjectInfo>* objects);

        static inline float intersection_area(const ObjectInfo& a, const ObjectInfo& b);
        static void qsort_descet_inplace(std::vector<ObjectInfo>& faceobjects, int left, int right);
        static void qsort_descent_inplace(std::vector<ObjectInfo>& faceobjects);
        static void nms_sorted_bboxes(const std::vector<ObjectInfo>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic=false);
        static inline float sigmoid(float x);
        static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<ObjectInfo>& objects);

    private:
        std::vector<ObjectInfo> objects;
        bool intialized_;
        ncnn::Net*  yolov5;
        const float nms_threshold = 0.6f;
        const float threshold_ = 0.8f;



    };
}

#endif //_YOLOV5_H
