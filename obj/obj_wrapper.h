//
// Created by throne on 19-5-12.
//

#ifndef HW2_OBJ_WRAPPER_H
#define HW2_OBJ_WRAPPER_H

#include <vector>
#include <string>
#include "../common/common.h"
#include "../common/geometry.hpp"

std::tuple<std::vector<utils::Vector3>, std::vector<TriangleFace>, std::vector<utils::Vector3>> loadObject(const std::string fname);
#endif //HW2_OBJ_WRAPPER_H
