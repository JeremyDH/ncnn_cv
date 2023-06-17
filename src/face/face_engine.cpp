#include "face_engine.h"
#include <iostream>

#include "detecter/detecter.h"
#include "landmarker/landmarker.h"

namespace mirror
{
    class FaceEngine::Impl{
    public:
        Impl() {
            detecter_factory_ = new RetinafaceFactory();
            detecter_factory_2 = new CenterfaceFactory();
            detecter_factory_3 = new MtcnnfaceFactory();
            landmarK_factory_ = new InsightfaceLandmarkerFactory();
            zq_landmark_factory_ = new ZQLandmarkerFacetory();

            detecter_ = detecter_factory_->CreateDetecter();
            landmarker_ = landmarK_factory_->CreateLandmarker();
            zq_ladmarker_ = zq_landmark_factory_->CreateLandmarker();
            detecter_2 = detecter_factory_2->CreateDetecter();
            detecter_3 = detecter_factory_3->CreateDetecter();

        }

        ~Impl(){
            if(detecter_){
                delete detecter_;
                detecter_ = nullptr;
            }
            if(detecter_factory_){
                delete detecter_factory_;
                detecter_factory_ = nullptr;
            }

            if(detecter_2){
                delete detecter_2;
                detecter_2 = nullptr;
            }
            if(detecter_factory_2){
                delete detecter_factory_2;
                detecter_factory_2 = nullptr;
            }

            if(detecter_3){
                delete detecter_3;
                detecter_3 = nullptr;
            }
            if(detecter_factory_3){
                delete detecter_factory_3;;
                detecter_factory_3 = nullptr;
            }

            if(landmarker_){
                delete landmarker_;
                landmarker_ = nullptr;
            }
            if(landmarK_factory_){
                delete landmarK_factory_;
                landmarK_factory_ = nullptr;
            }

            if(zq_ladmarker_)
            {
                delete zq_ladmarker_;
                zq_ladmarker_ = nullptr;
            }

            if(zq_landmark_factory_)
            {
                delete zq_landmark_factory_;
                zq_landmark_factory_ = nullptr;
            }
        }

        int LoadModel(const char* root_path)
        {
            if(detecter_->LoadModel(root_path) != 0)
            {
                std::cout << "load face detecter failed" << std::endl;
                return 10000;
            }
            if(detecter_2 ->LoadModel(root_path) != 0)
            {
                std::cout << "load face detecter failed" << std::endl;
                return 10000;
            }
            if(detecter_3 ->LoadModel(root_path) !=0 )
            {
                std::cout << "load face detecter3 failed" << std::endl;
                return 10000;
            }

            if(landmarker_->LoadModel(root_path) != 0)
            {
                std::cout << "load landmarker_ failed." << std::endl;
                return 10000;
            }

            if(zq_ladmarker_->LoadModel(root_path) != 0)
            {
                std::cout << "load zq_ladmarker_ failed." << std::endl;
                return 10000;
            }

            db_name_ = std::string(root_path);
            initialized_ = true;

            return 0;
        }

        inline int DetectFace(const cv::Mat& img_src, std::vector<FaceInfo>* faces){
            return detecter_2 ->DetectFace(img_src, faces);
        }

        inline int ExtractKeypoints(const cv::Mat& img_src,
                                   const cv::Rect& face, std::vector<cv::Point2f>* keypoints){
//            return landmarker_->ExtractKeypoints(img_src, face, keypoints);
              return zq_ladmarker_->ExtractKeypoints(img_src, face, keypoints);
        }


    private:
        DetecterFactory* detecter_factory_ = nullptr;
        InsightfaceLandmarkerFactory*  landmarK_factory_ = nullptr;
        ZQLandmarkerFacetory* zq_landmark_factory_ = nullptr;
        CenterfaceFactory*  detecter_factory_2 = nullptr;
        MtcnnfaceFactory* detecter_factory_3 = nullptr;


    private:
        bool initialized_;
        std::string db_name_;
        Detecter* detecter_ = nullptr;
        Detecter* detecter_2 = nullptr;
        Detecter* detecter_3 = nullptr;
        Landmarker* landmarker_ = nullptr;
        Landmarker* zq_ladmarker_= nullptr;

    };

    FaceEngine::FaceEngine() {
        impl_ = new FaceEngine::Impl();
    }

    FaceEngine::~FaceEngine() {
        if(impl_) {
            delete impl_;
            impl_ = nullptr;
        }
    }

    int FaceEngine::LoadModel(const char *root_path) {
        return impl_->LoadModel(root_path);
    }

    int FaceEngine::DetectFace(const cv::Mat &img_src, std::vector<FaceInfo> *faces) {
        return impl_->DetectFace(img_src, faces);
    }

    int FaceEngine::ExtractKeypoints(const cv::Mat& img_src, const cv::Rect& face,
                                     std::vector<cv::Point2f> *keypoints) {
        return  impl_->ExtractKeypoints(img_src, face, keypoints);
    }
}