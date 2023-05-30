//
// Created by dWX1185603 on 2023/5/30.
//
#include "detecter.h"
#include "retinaface/retinaface.h"

namespace mirror
{
    Detecter* RetinafaceFactory::CreateDetecter() {
        return new RetinaFace();
    }
}