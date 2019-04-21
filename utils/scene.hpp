#pragma once

#include "object.hpp"
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

    // find the first obj that the ray intersects(with minimum t)
    std::tuple<int, double, Point2D> findFirstIntersect(const Ray &r) const {
        double t = INF;
        size_t id = -1;
        Point2D param(0, 0);
        for (size_t i = 0; i < objects.size(); ++i) {
            auto res = objects[i]->intersect(r);
            if (std::get<1>(res) < t && std::get<0>(res).len2() > EPSILON)
                t = std::get<1>(res), id = i, param = std::get<2>(res);
        }
        return {id, t, param};
    }

    BasicObject *object(size_t idx) const {
        if (idx >= objects.size()) {
            return nullptr;
        }
        return objects[idx];
    }
};