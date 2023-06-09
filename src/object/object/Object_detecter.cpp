//
// Created by dWX1185603 on 2023/6/8.
//

#include "Object_detecter.h"
#include "mobilenetssd/mobilenetssd.h"

namespace mirror{

    object_detecter* MobilenetssdFactory::CreateMobilenetssd()
    {
            return new mobilenetssd();
    }

}