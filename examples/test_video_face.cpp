//
// Created by dWX1185603 on 2023/6/6.
//
//
// Created by dWX1185603 on 2023/5/30.
//
#define FACE_EXPORTS
//#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "../src/face/face_engine.h"
#include "iostream"

using namespace mirror;


void image_deal(FaceEngine* face_engine ,cv::Mat& img_src)
{
    std::vector<FaceInfo> faces;
    face_engine->DetectFace(img_src, &faces);
    for(int i=0; i<static_cast<int>(faces.size()); i++)
    {
        cv::Rect face = faces.at(i).location_;
        std::vector<cv::Point2f> keypoints;
        face_engine->ExtractKeypoints(img_src, face, &keypoints);
        for(int j=0; j<static_cast<int>(keypoints.size()); j++)
        {
            cv::circle(img_src, keypoints[j], 1, cv::Scalar(0, 0, 255), 1);
        }
        cv::rectangle(img_src, face, cv::Scalar(0, 255, 0), 2);
    }

}

//视频读写函数
int video_w(FaceEngine* face_engine)
{
    //读取视频
    const char* video_file = "D:\\coding_learning\\ncnn_example\\data\\video\\more_people.mp4";
    cv::VideoCapture capture(video_file);
//    capture.open("D:\\coding_learning\\ncnn_example\\data\\video\\more_people.mp4", cv::CAP_FFMPEG);
    if (!capture.isOpened()){
        printf("could not read this video file...\n");
        return -1;
    }
    cv::Size S = cv::Size((int) capture.get(cv::CAP_PROP_FRAME_WIDTH),
                          (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = capture.get(cv::CAP_PROP_FPS);
    printf("current fps: %d \n", fps);

//  创建写入对象
    cv::VideoWriter writer("./data/video/test_face.avi", cv::CAP_OPENCV_MJPEG, fps, S, true);

    cv::Mat frame;
    cv::namedWindow("video_face", cv::WINDOW_AUTOSIZE);
    while(capture.read(frame)){
        image_deal(face_engine, frame);
        writer.write(frame);

    }
    return 0;

}


int TestDetecter()
{
    const char* root_path = "./model";

    double start = static_cast<double>(cv::getTickCount());


    FaceEngine* face_engine = new FaceEngine();
    face_engine->LoadModel(root_path);

    video_w(face_engine);

//    face_engine->DetectFace(img_src, &faces);


    double end = static_cast<double>(cv::getTickCount());
    double time_cost = (end - start) / cv::getTickFrequency() * 1000;
//    std::cout << "time cost: " << time_cost << "ms" << std::endl;
//    cv::imwrite("data/images/result.jpg", img_src);
//    cv::imshow("result", img_src);
//    cv::waitKey(0);

    delete face_engine;
    face_engine = nullptr;

    std::cout << "using time: " << time_cost << std::endl;

    return 0;

}

int main(int argc, char* argv[])
{
    //加载视频文件
    return TestDetecter();
}