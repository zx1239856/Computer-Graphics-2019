//
// Created by zx on 19-4-27.
//

#ifndef HW2_MATH_HELPERS_H
#define HW2_MATH_HELPERS_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include "../common/common.h"
#include "../common/geometry.hpp"

__host__ __device__ inline double clamp(double v, double low = 0.0, double high = 1.0) {
    return fmin(fmax(v, low), high);
}

inline int toUInt8(double x) { int v = int(pow(clamp(x), 1 / 2.2) * 255 + .5); return v > 255 ? 255 : v; }

__host__ __device__ inline utils::Vector3 uniformSampleOnHemisphere(double u1, double u2) {
    double sin_theta = sqrt(fmax(.0, 1.0 - u1 * u1));
    double phi = 2. * M_PI * u2;
    return utils::Vector3(cos(phi) * sin_theta, sin(phi) * sin_theta, u1);
}

__host__ __device__ inline utils::Vector3 cosWeightedSampleOnHemisphere(double u1, double u2) {
    double cos_theta = sqrt(1. - u1);
    double sin_theta = sqrt(u1);
    double phi = 2. * M_PI * u2;
    return utils::Vector3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

#endif //HW2_MATH_HELPERS_H
