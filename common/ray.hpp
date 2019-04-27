#pragma once
#include "geometry.hpp"
#include <stdio.h>

#ifndef __NVCC__
#define __device__
#define __host__
#endif

class Ray
{
public:
    utils::Vector3 origin, direction;
    __device__ __host__ Ray(utils::Vector3 o, utils::Vector3 d): origin(o), direction(d) {}
    __device__ __host__ utils::Vector3 getVector(double t)const{ return origin + direction * t; }
};