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