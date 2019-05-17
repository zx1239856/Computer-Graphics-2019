/*
 * Created by zx on 19-4-1.
 */

#ifndef HW2_RENDERER_H
#define HW2_RENDERER_H

#include "../common/geometry.hpp"
#include "../common/ray.hpp"
#include "hitPoint.hpp"

class Scene;

utils::Vector3 basic_pt(const Scene & scene, const Ray &ray, int depth, unsigned short *X, bool ray_cast = false);

void ray_trace(const Scene &scene, const Ray &ray, const utils::Vector3 &flux, const utils::Vector3 &weight, int depth, unsigned short *X,
               HitPointKDTree &tree, const bool &cam_pass, const uint32_t &pixel_place);

void evalRadiance(std::vector<utils::Vector3> &out, const utils::Vector3 &light, const int w, const int h, const HitPointKDTree &tree,
                  const int num_rounds, const int num_photons, const int super_sampling);

void evalRadianceTest(std::vector<utils::Vector3> &out, const int w, const int h, const HitPointKDTree &tree);

#endif //HW2_RENDERER_H
