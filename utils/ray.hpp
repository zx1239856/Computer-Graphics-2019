#pragma once
#include "vector3.hpp"
#include <stdio.h>

class Ray
{
public:
    utils::Vector3 origin, direction;
    Ray(utils::Vector3 o, utils::Vector3 d): origin(o), direction(d) {}
    utils::Vector3 getVector(double t)const{ return origin + direction * t; }
};