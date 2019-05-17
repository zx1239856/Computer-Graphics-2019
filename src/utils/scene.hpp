#pragma once

#include "object.hpp"
#include <stdio.h>

class Scene {
    std::vector<BasicObject *> objects;
    Ray light;  // for sppm use
public:
    Scene() : light(utils::Vector3(), utils::Vector3()) {}

    ~Scene() {
        for (auto x: objects) {
            if (x)delete x;
        }
    }

    Ray genRay(unsigned short *X) const{
        Ray ray = light;
        // change dir
        double r1 = 2 * M_PI * erand48(X), r2s = sqrt(erand48(X));
        utils::Vector3 w = light.direction, u = ((fabs(w.x()) > .1 ? utils::Vector3(0, 1) : utils::Vector3(1)).cross(
                light.direction)).normalize(), v = w.cross(u).normalize();
        ray.direction = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2s * r2s)).normalize();
        return ray;
    }

    void setLight(const utils::Vector3 &src, const utils::Vector3 &dir = utils::Vector3(0, -1, 0)) {
        light.origin = src;
        light.direction = dir.normalize();
    }

    void addObject(BasicObject *obj) { objects.emplace_back(obj); }

    // find the first obj that the ray intersects(with minimum t)
    std::tuple<int, double, utils::Point2D, utils::Vector3> findFirstIntersect(const Ray &r) const {
        double t = INF;
        size_t id = -1;
        utils::Point2D param(0, 0);
        utils::Vector3 norm;
        for (size_t i = 0; i < objects.size(); ++i) {
            auto res = objects[i]->intersect(r);
            if (std::get<1>(res) < t && std::get<1>(res) > EPSILON)
                t = std::get<1>(res), id = i, param = std::get<2>(res), norm = std::get<0>(res);
        }
        return {id, t, param, norm};
    }

    BasicObject *object(size_t idx) const {
        if (idx >= objects.size()) {
            return nullptr;
        }
        return objects[idx];
    }
};