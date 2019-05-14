//
// Created by zx on 19-4-27.
//
#ifndef HW2_SCENE_H
#define HW2_SCENE_H

#include <cuda_runtime.h>
#include "cuda_helpers.h"
#include "../utils/imghelper.h"

struct __align__(16) Camera{
        int w, h;
        utils::Vector3 origin, direction;
        double fov, aperture, focal_dist; // camera intrincs
};

__device__ inline triplet<utils::Vector3, double, pair<utils::Point2D, Texture_GPU*>>
findFirstIntersect(const KernelArray<Plane_GPU> &planes, const KernelArray<Cube_GPU> &cubes,
                   const KernelArray<Sphere_GPU> &spheres, const KernelArray<RotaryBezier_GPU> &beziers,
                   const KernelArray<TriangleMeshObject_GPU> &meshes,
                   const Ray &r) {
    double t = INF;
    utils::Point2D param(0, 0);
    Texture_GPU *texture = nullptr;
    utils::Vector3 norm;
    for (size_t i = 0; i < planes._size; ++i) {
        auto res = planes._array[i].intersect(r);
        if (res.second < t)
            t = res.second, param = res.third, texture = &planes._array[i].texture, norm = res.first;
    }
    for (size_t i = 0; i < cubes._size; ++i) {
        auto res = cubes._array[i].intersect(r);
        if (res.second < t)
            t = res.second, param = res.third, texture = &cubes._array[i].texture, norm = res.first;
    }
    for (size_t i = 0; i < spheres._size; ++i) {
        auto res = spheres._array[i].intersect(r);
        if (res.second < t)
            t = res.second, param = res.third, texture = &spheres._array[i].texture, norm = res.first;
    }
    for (size_t i = 0; i < beziers._size; ++i) {
        auto res = beziers._array[i].intersect(r);
        if (res.second < t)
            t = res.second, param = res.third, texture = &beziers._array[i].texture, norm = res.first;
    }
    for (size_t i = 0; i < meshes._size; ++i) {
        auto res = meshes._array[i].intersect(r);
        if(res.second < t)
            t = res.second, param = res.third, texture = &meshes._array[i].texture, norm = res.first;
    }
    return {norm, t, {param, texture}};
}

#endif //HW2_SCENE_H
