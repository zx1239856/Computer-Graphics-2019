/*
 * Created by zx on 19-4-1.
 */

#ifndef HW2_RENDERER_H
#define HW2_RENDERER_H

#include "geometry.hpp"
#include "ray.hpp"

class Scene;

utils::Vector3 basic_pt(const Scene & scene, const Ray &ray, int depth, unsigned short *X);

#endif //HW2_RENDERER_H
