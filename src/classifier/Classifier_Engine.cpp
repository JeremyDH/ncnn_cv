//
// Created by dWX1185603 on 2023/6/9.
//


#include "Classifier_Engine.h"
#include "./classifier/Classifier_Detecter.h"

namespace mirror{
    class Classifier_Engine::Impl_{
    public:
        Impl_(){

            mobilenet_factory = new Mobilenet_Factory();

            classifier_detecter = mobilenet_factory->createClassifier();

        }

        ~Impl_(){

            if(mobilenet_factory)
            {
                delete mobilenet_factory;
                mobilenet_factory = nullptr;
            }

            if(classifier_detecter)
            {
                delete classifier_detecter;
                classifier_detecter = nullptr;
            }

        }

        int Loadmodel(const char* root_path)
        {
            if(classifier_detecter->Loadmodel(root_path) != 0){
                std::cout << "load model failed"  << std::endl;
                return 10000;
            }
            db_name_ = std::string(root_path);
            initialized_ = true;
            return 0;

        }

        inline int Classifier_de(cv::Mat& img_src, std::vector<ImageInfo>* objectes)
        {
             return classifier_detecter->Class_Detect(img_src, objectes);

        }

    private:
        Mobilenet_Factory*  mobilenet_factory = nullptr;


    private:
        std::string db_name_;
        bool initialized_;
        Classifier_Detecter* classifier_detecter = nullptr;

    };

    Classifier_Engine::Classifier_Engine() {
        impl_ = new Impl_();
    }
    Classifier_Engine::~Classifier_Engine() {
        if(impl_){
            delete impl_;
            impl_ = nullptr;
        }
    }

    int Classifier_Engine::LoadModel(const char *root_path) {
         return  impl_->Loadmodel(root_path);
    }
    int Classifier_Engine::Classifier(cv::Mat &img_src, std::vector<ImageInfo>* classifers) {
         return impl_->Classifier_de(img_src, classifers);

    }
}