//
// Created by Jeremy Dong on 2023/6/5.
//
#define FACE_EXPORTS
#include "opencv2/opencv.hpp"
#include "../src/face/face_engine.h"

using namespace mirror;

int TestDetecter(int argc, char* argv[]) {
    const char* img_file = "data/messi.jpg";
    cv::Mat img_src = cv::imread(img_file);
    const char* root_path = "model";

    FaceEngine* face_engine = new FaceEngine();
    face_engine->LoadModel(root_path);
    std::vector<FaceInfo> faces;
    double start = static_cast<double>(cv::getTickCount());
    face_engine->DetectFace(img_src, &faces);
    double end = static_cast<double>(cv::getTickCount());
    double time_cost = (end - start) / cv::getTickFrequency() * 1000;
    std::cout << "time cost: " << time_cost << "ms" << std::endl;

    for (int i = 0; i < static_cast<int>(faces.size()); ++i) {
        FaceInfo face_info = faces.at(i);
        cv::rectangle(img_src, face_info.location_, cv::Scalar(0, 255, 0), 2);
#if 1
        for (int num = 0; num < 5; ++num) {
            cv::Point curr_pt = cv::Point(face_info.keypoints_[num],
                                          face_info.keypoints_[num + 5]);
            cv::circle(img_src, curr_pt, 2, cv::Scalar(255, 0, 255), 2);
        }
#endif
    }
    cv::imwrite("data/retinaface_result.jpg", img_src);
    cv::imshow("result", img_src);
    cv::waitKey(0);

    delete face_engine;
    face_engine = nullptr;

    return 0;
}

int TestLandmark(int argc, char* argv[])
{
    std::string img_path = "data/messi2.jpg";
    cv::Mat img_src = cv::imread(img_path);

    const char* root_path = "model";

    double start_time = static_cast<double>(cv::getTickCount());

    FaceEngine* face_engine = new FaceEngine();
    face_engine->LoadModel(root_path);

    std::vector<FaceInfo> faces;
    face_engine->DetectFace(img_src, &faces);

    for(int i = 0; i < static_cast<int>(faces.size()); i++)
    {
        cv::Rect face = faces.at(i).location_;
        std::vector<cv::Point2f> keypoints;
        face_engine->ExtractKeypoints(img_src, face, &keypoints);
        for(int j = 0; j < static_cast<int>(keypoints.size()); j++)
        {
            cv::circle(img_src, keypoints[j], 1, cv::Scalar(0, 0, 255), 1);
        }
        cv::rectangle(img_src, face, cv::Scalar(0, 255, 0), 2);
    }
    double end_time = static_cast<double>(cv::getTickCount());
    double time_cast = (end_time - start_time) / cv::getTickFrequency() * 1000;
    std::cout << "using time: " << time_cast << std::endl;
    cv::imwrite("data/result.jpg", img_src);
    cv::imshow("result", img_src);
    cv::waitKey(0);


    delete face_engine;
    face_engine = nullptr;
    return 0;

}


int main(int argc, char* argv[])
{
//    return TestDetecter(argc, argv);
    TestLandmark(argc,  argv);

    return 0;

}