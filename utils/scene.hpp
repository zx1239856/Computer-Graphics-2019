#pragma once

#include "object.hpp"
#include "camera.hpp"
#include "renderer.h"
#include <stdio.h>

class Scene {
    std::vector<BasicObject *> objects;
public:
    Scene() {}

    ~Scene() {
        for (auto x: objects) {
            if (x)delete x;
        }
    }

    void addObject(BasicObject *obj) { objects.emplace_back(obj); }

    std::vector<utils::Vector3> render(const Camera &cam, int samps, int w, int h) const {
        std::vector<utils::Vector3> c(w * h);
        utils::Vector3 cx = utils::Vector3(w * .4135 / h), cy = cx.cross(cam.direction).normalize() * .4135, r;
#pragma omp parallel for schedule(dynamic, 1) private(r)
        for (int y = 0; y < h; ++y) { // rows
            fprintf(stderr, "\rRendering (%d spp) %5.2f%%", samps * 4, 100. * y / (h - 1));
            for (ushort x = 0, X[3] = {0, 0, static_cast<ushort>(y * y * y)}; x < w; ++x) // cols
                for (int sy = 0, i = (h - y - 1) * w + x; sy < 2; ++sy)
                    for (int sx = 0; sx < 2; ++sx, r = utils::Vector3()) {
                        for (int s = 0; s < samps; ++s) {
                            double r1 = 2 * erand48(X), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
                            double r2 = 2 * erand48(X), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
                            utils::Vector3 d = cx * (((sx + .5 + dx) / 2 + x) / w - .5) +
                                               cy * (((sy + .5 + dy) / 2 + y) / h - .5) + cam.direction;
                            d = d.normalize();
                            r = r + basic_pt(*this, Ray(cam.origin, d), 0, X) * (1. / samps);
                        }
                        c[i] = c[i] + r.clamp(0, 1) * .25;
                    }
        }
        return c;
    }

    // find the first obj that the ray intersects(with minimum t)
    std::pair<int, double> findFirstIntersect(const Ray &r) const {
        double t = INF;
        size_t id = -1;
        for (size_t i = 0; i < objects.size(); ++i) {
            auto res = objects[i]->intersect(r);
            if (res.second < t && res.first.len2() > epsilon)
                t = res.second, id = i;
        }
        return {id, t};
    }

    BasicObject *object(size_t idx) const { return objects[idx]; }
};