//
// Created by dWX1185603 on 2023/5/30.
//
#include "detecter.h"
#include "retinaface/retinaface.h"
#include "centerface/centerface.h"
#include "mtcnn/mtcnn.h"

namespace mirror
{
    Detecter* RetinafaceFactory::CreateDetecter() {
        return new RetinaFace();
    }
    Detecter* CenterfaceFactory::CreateDetecter() {
        return new Centerface();
    }
    Detecter* MtcnnfaceFactory::CreateDetecter() {
        return new Mtcnnface();
    }
}