#pragma once
#include "vector3.hpp"

class Camera
{
public:
    utils::Vector3 origin, direction;
    Camera(const utils::Vector3 &o, const utils::Vector3 &d): origin(o), direction(d) {}
};