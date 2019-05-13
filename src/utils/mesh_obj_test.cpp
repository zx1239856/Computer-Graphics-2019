//
// Created by throne on 19-5-12.
//
#include "../obj/obj_wrapper.h"
#include "object.hpp"
#include "../common/ray.hpp"
#include <iostream>

int main() {
    using namespace std;
    using namespace utils;
    auto param = loadObject("../model/angel_lucy.obj");
    TriangleMeshObject mesh(utils::Vector3(380, -1.19, 30), 0.1, std::get<0>(param), std::get<1>(param), std::get<2>(param),
                            Vector3(.8, .8, .8), Vector3(), BRDFs[DIFFUSE]);
    auto aabb = mesh.boundingBox();
    cout << "Bounding box of the mesh obj is: p0(" << aabb.first.x() << ", " << aabb.first.y() << ", " << aabb.first.z()
         << "), p1(" <<
         aabb.second.x() << ", " << aabb.second.y() << ", " << aabb.second.z() << ")" << endl;
    srand(time(nullptr));
    auto est = Vector3(37.88, -9.9, -37.65);
    for (int i = 0; i < 100; ++i) {
        cout << "#Round " << i << ":\t";
        auto direction = Vector3(rand() * 1. / RAND_MAX, rand() * 1. / RAND_MAX, rand() * 1. / RAND_MAX).normalize();
        Ray r(est + direction * rand(), -direction);
        auto res = mesh.intersect(r);
        if (get<1>(res) < INF) {
            auto pos = r.getVector(get<1>(res));
            cout << "The hit point is: ( " << pos.x() << ", " << pos.y() << ", " << pos.z() << ")" << endl;
        }
        else
            cout << "No intersection" << endl;
    }
    return 0;
}
