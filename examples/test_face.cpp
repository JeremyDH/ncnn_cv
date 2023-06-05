//
// Created by Jeremy Dong on 2023/6/5.
//
#define FACE_EXPORTS
#include "opencv2/opencv.hpp"
#include "../src/face/face_engine.h"

int TestDetecter(int argc, char* argv[]) {
    const char* img_file = "../data/foot1.jpg";
    cv::Mat img_src = cv::imread(img_file);
    const char* root_path = "../model";

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
    cv::imwrite("../../data/images/retinaface_result.jpg", img_src);
    cv::imshow("result", img_src);
    cv::waitKey(0);

    delete face_engine;
    face_engine = nullptr;

    return 0;
}


int main(int argc, char* argv[])
{
    return TestDetecter(argc, argv);

}