//
// Created by dWX1185603 on 2023/6/2.
//

#include "landmarker.h"
#include "insightface/insightface.h"
#include "zqlandmarker/zqlandmarker.h"

namespace mirror{
    Landmarker* InsightfaceLandmarkerFactory::CreateLandmarker() {
        return new Insightface();
    }

    Landmarker* ZQLandmarkerFacetory::CreateLandmarker() {
        return new ZQLandmarker();
    }
}
