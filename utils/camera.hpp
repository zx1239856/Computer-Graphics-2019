#pragma once

#include "vector3.hpp"
#include "scene.hpp"
#include "renderer.h"

class Camera {
private:
    int w, h;
    utils::Vector3 origin, direction;
    double fov, aperture, focal_dist; // camera intrincs
    Scene *scene;
public:
    Camera(int _w, int _h) : w(_w), h(_h), scene(nullptr) {}

    void setResolution(int w, int h) {
        this->w = w;
        this->h = h;
    }

    void setPosition(const utils::Vector3 &origin, const utils::Vector3 &direction) {
        this->origin = origin;
        this->direction = direction;
    }

    void setLensParam(double fov, double aperture, double focal_dist) {
        this->fov = fov;
        this->aperture = aperture;
        this->focal_dist = focal_dist;
    }

    void setScene(Scene *scene) {
        this->scene = scene;
    }

    std::vector<utils::Vector3> renderPt(int samps) {
        std::vector<utils::Vector3> c(w * h);
        if (scene == nullptr)
            return c;
        utils::Vector3 cx = utils::Vector3(w * fov / h), cy = cx.cross(direction).normalize() * fov, r;
#pragma omp parallel for schedule(dynamic, 1) private(r)
        for (int v = 0; v < h; ++v) {
            fprintf(stderr, "\rRendering (%d spp) %5.2f%%", samps * 4, 100. * v / (h - 1));
            for (int u = 0; u < w; ++u) {
                for (int sy = 0; sy < 2; ++sy) {  // 2x2 super sampling
                    for(int sx = 0; sx < 2; ++sx) {
                        ushort X[3] = {static_cast<ushort>(v + sx), static_cast<ushort>(v * u + sy), static_cast<ushort>(v * u * v + sx * sy)};
                        r = utils::Vector3();
                        for (int s = 0; s < samps; ++s) {
                            double r1 = 2 * erand48(X), dx = r1 < 1 ? sqrt(r1) : 2 - sqrt(2 - r1);
                            double r2 = 2 * erand48(X), dy = r2 < 1 ? sqrt(r2) : 2 - sqrt(2 - r2);
                            utils::Vector3 d = cx * (((sx + .5 + dx) / 2 + u) / w - .5) +
                                               cy * (((sy + .5 + dy) / 2 + v) / h - .5) + direction;
                            utils::Vector3 hit = origin + d * focal_dist;  // real hit point on focal plane
                            utils::Vector3 p_origin = origin + (utils::Vector3(erand48(X) * 1.01,erand48(X)) - .5) * 2 * aperture; // origin perturbation
                            d = d.normalize();
                            r = r + basic_pt(*scene, Ray(p_origin, (hit - p_origin).normalize()), 0, X) * (1. / samps);
                        }
                        c[(h - v - 1) * w + u] += r.clamp(0, 1) * .25;
                    }
                }
            }
        }
        return c;
    }
};