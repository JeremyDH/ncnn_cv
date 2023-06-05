#include "face_engine.h"
#include <iostream>

#include "detecter/detecter.h"
#include "landmarker/landmarker.h"



namespace mirror {
    class FaceEngine::Impl {
    public:
        Impl() {
            // detecter_factory_ = new AnticonvFactory();
            detecter_factory_ = new RetinafaceFactory();
            landmarker_factory_ = new InsightfaceLandmarkerFactory();


            detecter_ = detecter_factory_->CreateDetecter();
            landmarker_ = landmarker_factory_->CreateLandmarker();


            initialized_ = false;
        }

        ~Impl() {
            if (detecter_) {
                delete detecter_;
                detecter_ = nullptr;
            }

            if (landmarker_) {
                delete landmarker_;
                landmarker_ = nullptr;
            }

            if (detecter_factory_) {
                delete detecter_factory_;
                detecter_factory_ = nullptr;
            }

            if (landmarker_factory_) {
                delete landmarker_factory_;
                landmarker_factory_ = nullptr;
            }

        }

        int LoadModel(const char* root_path) {
            if (detecter_->LoadModel(root_path) != 0) {
                std::cout << "load face detecter failed." << std::endl;
                return 10000;
            }

            if (landmarker_->LoadModel(root_path) != 0) {
                std::cout << "load face landmarker failed." << std::endl;
                return 10000;
            }

            db_name_ = std::string(root_path);
            initialized_ = true;

            return 0;
        }

        inline int DetectFace(const cv::Mat& img_src, std::vector<FaceInfo>* faces) {
            return detecter_->DetectFace(img_src, faces);
        }
        inline int ExtractKeypoints(const cv::Mat& img_src,
                                    const cv::Rect& face, std::vector<cv::Point2f>* keypoints) {
            return landmarker_->ExtractKeypoints(img_src, face, keypoints);
        }



    private:
        DetecterFactory* detecter_factory_ = nullptr;
        LandmarkerFactory* landmarker_factory_ = nullptr;

    private:
        bool initialized_;
        std::string db_name_;
        Detecter* detecter_ = nullptr;
        Landmarker* landmarker_ = nullptr;
    };

    FaceEngine::FaceEngine() {
        impl_ = new FaceEngine::Impl();
    }

    FaceEngine::~FaceEngine() {
        if (impl_) {
            delete impl_;
            impl_ = nullptr;
        }
    }

    int FaceEngine::LoadModel(const char* root_path) {
        return impl_->LoadModel(root_path);
    }

    int FaceEngine::DetectFace(const cv::Mat& img_src, std::vector<FaceInfo>* faces) {
        return impl_->DetectFace(img_src, faces);
    }

    int FaceEngine::ExtractKeypoints(const cv::Mat& img_src,
                                     const cv::Rect& face, std::vector<cv::Point2f>* keypoints) {
        return impl_->ExtractKeypoints(img_src, face, keypoints);
    }


}