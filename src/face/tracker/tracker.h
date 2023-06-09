//
// Created by dWX1185603 on 2023/6/8.
//

#ifndef _FACE_TRACKER_H
#define _FACE_TRACKER_H
#include <vector>
#include "../../common/common.h"
#include "opencv2/opencv.hpp"

namespace mirror {
    class Tracker {
    public:
        Tracker();
        ~Tracker();
        int Track(const std::vector<FaceInfo>& curr_faces,
                  std::vector<TrackedFaceInfo>* faces);
    private:
        std::vector<TrackedFaceInfo> pre_tracked_faces_;
        const float mixScore_ = 0.3f;
        const float maxScore_ = 0.5f;
    };
}

#endif // _TRACKER_H
