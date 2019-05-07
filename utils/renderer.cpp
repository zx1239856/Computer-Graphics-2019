#include "renderer.h"
#include "scene.hpp"
#include "object.hpp"

utils::Vector3 basic_pt(const Scene &scene, const Ray &ray, int depth, unsigned short *X) {
    using namespace utils;
    auto intersect = scene.findFirstIntersect(ray);
    if (std::get<0>(intersect) == -1)
        return utils::Vector3(); // no intersection
    BasicObject *obj = scene.object(std::get<0>(intersect));
    if(obj == nullptr)
        return utils::Vector3();
    bool into = false;
    utils::Vector3 x = ray.getVector(std::get<1>(intersect)), n = obj->norm(x, std::get<2>(intersect)), nl =
            n.dot(ray.direction) < 0 ? into = true, n : -n;
    const Texture& texture = obj->texture;
    auto && f = texture.getColor(std::get<2>(intersect), X);
    double p = f.first.max();
    if (p < EPSILON)
        return texture.emission;
    if (++depth > 5) {
        if (erand48(X) < p)f.first = f.first * (1 / p);
        else return texture.emission;
    }
    if (f.second == DIFF) {
        double a = erand48(X);   // phong model
        double s = 1;
        if(a < texture.rho_s)
            s = texture.phong_s;
        if(a >= texture.rho_d)
            return texture.emission;
        double r1 = 2 * M_PI * erand48(X), r2s = (pow(erand48(X), 1./(s + 1)));
        Vector3 w = nl, u = ((fabs(w.x()) > .1 ? Vector3(0, 1) : Vector3(1)).cross(nl)).normalize(), v = w.cross(u);
        Vector3 d = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2s * r2s)).normalize();
        return texture.emission + f.first.mult(basic_pt(scene, Ray(x, d), depth, X));
    } else if (f.second == SPEC)
        return texture.emission + f.first.mult(basic_pt(scene, Ray(x, ray.direction.reflect(nl)), depth, X));
    else {
        Ray reflray = Ray(x, ray.direction.reflect(nl));
        Vector3 d = ray.direction.refract(n, into ? 1 : texture.re_idx, into ? texture.re_idx : 1);
        if (d.len2() < EPSILON) // total internal reflection
            return texture.emission + f.first.mult(basic_pt(scene, reflray, depth, X));
        double a = texture.re_idx - 1, b = texture.re_idx + 1, R0 = a * a / (b * b), c =
                1 - (into ? -ray.direction.dot(nl) : d.dot(n));
        double Re = R0 + (1 - R0) * c * c * c * c * c, Tr = 1 - Re, P = .25 + .5 * Re, RP = Re / P, TP = Tr / (1 - P);
        return texture.emission + f.first.mult(depth > 2 ? (erand48(X) < P ?
                                                      basic_pt(scene, reflray, depth, X) * RP :
                                                      basic_pt(scene, Ray(x, d), depth, X) * TP) :
                                         basic_pt(scene, reflray, depth, X) * Re +
                                         basic_pt(scene, Ray(x, d), depth, X) * Tr);
    }
}