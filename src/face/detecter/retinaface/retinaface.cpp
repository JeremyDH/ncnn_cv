//
// Created by dWX1185603 on 2023/5/30.
//
#include "retinaface.h"
#include <iostream>
#if MIRROR_VULKAN
#include "gpu.h"
#endif //MIRROR_VULKAN

namespace mirror{
    RetinaFace::RetinaFace():
    retina_net_(new ncnn::Net()),
    initialized_(false){
#if MIRROR_VULKAN
        ncnn::create_gpu_instance();
    retina_net_->opt.use_vulkan_compute = true;
#endif // MIRROR_VULKAN
    }

    RetinaFace::~RetinaFace() {
        if(retina_net_){
            retina_net_->clear();
        }
#if MIRROR_VULKAN
        ncnn::destroy_gpu_instance();
#endif //MIRROR_VULKAN
    }


}