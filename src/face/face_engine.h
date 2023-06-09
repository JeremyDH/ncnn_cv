//
// Created by dWX1185603 on 2023/5/30.
//

#ifndef _FACE_ENGINE_H_
#define _FACE_ENGINE_H_

#include <vector>
#include "../common/common.h"
#include "opencv2/core.hpp"


/**
 * dllexport和 dllimport都是DLL内的关键字，即导入和导出。
 * dllexport是在这些类、函数以及数据的申明的时候使用。用他表明这些东西可以被外部函数使用，
 * dllexport是把DLL中的相关代码（类、函数、数据）暴露出来为其他程序使用
 *
 * dllimport是在外部程序需要使用DLL内相关内容时使用的关键字。当一个外部程序要使用DLL内部代码（类、函数、全局变量）时，
 * 只需要在程序内部使用dllimport关键字声明即可。
 *
 * _declspec(dllexport)与_declspec(dllimport)是相互呼应，只有在DLL内部用dllexport作了声明，才能在外部函数中用dllimport导入相关的代码。
 *
 * **/

#if defined(_MSC_VER) || defined(_WIN32) || defined(_WIN64)
#ifdef FACE_EXPORTS
#define FACE_API __declspec(dllexport)
#else
#define FACE_API __declspec(dllimport)
#endif
#else
#define FACE_API __attribute__ ((visibility("default")))
#endif

namespace mirror {
    class FaceEngine {
    public:
        FACE_API FaceEngine();
        FACE_API ~FaceEngine();
        FACE_API int LoadModel(const char* root_path);
        FACE_API int DetectFace(const cv::Mat& img_src, std::vector<FaceInfo>* faces);
        FACE_API int Track(const std::vector<FaceInfo>& curr_faces,
                           std::vector<TrackedFaceInfo>* faces);
        FACE_API int ExtractKeypoints(const cv::Mat& img_src,
                                      const cv::Rect& face, std::vector<cv::Point2f>* keypoints);
        FACE_API int ExtractFeature(const cv::Mat& img_face, std::vector<float>* feature);
        FACE_API int AlignFace(const cv::Mat& img_src, const std::vector<cv::Point2f>& keypoints, cv::Mat* face_aligned);

        // database operation
        FACE_API int Insert(const std::vector<float>& feat, const std::string& name);
        FACE_API int Delete(const std::string& name);
        FACE_API int64_t QueryTop(const std::vector<float>& feat, QueryResult *query_result = nullptr);
        FACE_API int Save();
        FACE_API int Load();

    private:
        class Impl;
        Impl* impl_;

    };

}


#endif //_FACE_ENGINE_H
