#pragma once
#include "vector3.hpp"

class Ray
{
public:
    utils::Vector3 origin, direction;
    Ray(utils::Vector3 o, utils::Vector3 d): origin(o), direction(o) {}
    utils::Vector3 getVector(double t)const{ return origin + direction * t; }
};