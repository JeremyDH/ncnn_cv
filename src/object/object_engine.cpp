//
// Created by dWX1185603 on 2023/6/8.
//

#include "object_engine.h"
#include <iostream>
#include <string>
#include "object/mobilenetssd/mobilenetssd.h"

namespace mirror{
    class Object_engine::Impl{
    public:
        Impl(){
               mobilenetssd_factory = new MobilenetssdFactory();

               ob_detecter = mobilenetssd_factory->CreateMobilenetssd();

        }
        ~Impl(){
            if(mobilenetssd_factory){
               delete mobilenetssd_factory;
               mobilenetssd_factory = nullptr;
            }
            if(ob_detecter){
                delete ob_detecter;
                ob_detecter = nullptr;
            }

        }

        int Loadmodel(const char *root_path) {

            if(ob_detecter->Loadmodel(root_path) !=0){
                std::cout << "load model failed" << std::endl;
                return 10000;
            }
            db_name_ = std::string(root_path);
            initialized_ = true;
            return 0;
        }

        inline int Object_Detect(const cv::Mat& img_src, std::vector<ObjectInfo>* objects)
        {
            return ob_detecter->Object_d(img_src, objects);
        }

    private:
        MobilenetssdFactory* mobilenetssd_factory = nullptr;

    private:
        bool initialized_;
        std::string db_name_;
        object_detecter* ob_detecter = nullptr;

    };

    Object_engine::Object_engine() {
        impl_ = new Object_engine::Impl();
    }
    Object_engine::~Object_engine() {
        if(impl_) {
            delete impl_;
            impl_ = nullptr;
        }
    }

    int Object_engine::Loadmodel(const char *root_path) {
        return impl_->Loadmodel(root_path);
    }

   int Object_engine::DetectObject(const cv::Mat &img_src, std::vector<ObjectInfo> *objects) {
        return  impl_->Object_Detect(img_src, objects);
    }

}
