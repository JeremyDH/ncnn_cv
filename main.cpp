
#include "iostream"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "ncnn/net.h"
#include <ncnn/


using namespace std;

int main()
{

    cv::Mat imgs = cv::imread("picture/car.png", cv::IMREAD_COLOR);
    cv::imshow("now_pic", imgs);
    cv::waitKey(0);
    cv::destroyAllWindows();


    return 0;
}