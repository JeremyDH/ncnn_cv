//
// Created by dWX1185603 on 2023/6/9.
//

#include "Classifier_Detecter.h"
#include "mobilenet/mobilenet.h"

namespace mirror{

    Classifier_Detecter* Mobilenet_Factory::createClassifier() {
        return new mobilenet();
    }
}
