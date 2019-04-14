#include "renderer.h"
#include "scene.hpp"
#include "object.hpp"

utils::Vector3 basic_pt(const Scene &scene, const Ray &ray, int depth, unsigned short *X) {
    using namespace utils;
    auto intersect = scene.findFirstIntersect(ray);
    if (std::get<0>(intersect) == -1)
        return utils::Vector3(); // no intersection
    BasicObject *obj = scene.object(std::get<0>(intersect));
    bool into = false;
    utils::Vector3 x = ray.getVector(std::get<1>(intersect)), n = obj->norm(x, std::get<2>(intersect)), nl =
            n.dot(ray.direction) < 0 ? into = true, n : -n;
    Texture texture = obj->getTexture();
    utils::Vector3 f = texture.color;
    double p = f.max();
    if (f.max() < epsilon)
        return texture.emission;
    if (++depth > 5)
        if (erand48(X) < p)f = f * (1 / p);
        else return texture.emission;
    if (texture.refl == DIFF) {
        double r1 = 2 * M_PI * erand48(X), r2 = erand48(X), r2s = sqrt(r2);
        Vector3 w = nl, u = ((fabs(w.x()) > .1 ? Vector3(0, 1) : Vector3(1)).cross(w)).normalize(), v = w.cross(u);
        Vector3 d = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).normalize();
        return texture.emission + f.mult(basic_pt(scene, Ray(x, d), depth, X));
    } else if (texture.refl == SPEC)
        return texture.emission + f.mult(basic_pt(scene, Ray(x, ray.direction.reflect(nl)), depth, X));
    else {
        Ray reflray = Ray(x, ray.direction.reflect(nl));
        Vector3 d = ray.direction.refract(n, into ? 1 : texture.brdf, into ? texture.brdf : 1);
        if (d.len2() < epsilon) // total internal reflection
            return texture.emission + f.mult(basic_pt(scene, reflray, depth, X));
        double a = texture.brdf - 1, b = texture.brdf + 1, R0 = a * a / (b * b), c =
                1 - (into ? -ray.direction.dot(nl) : d.dot(n));
        double Re = R0 + (1 - R0) * c * c * c * c * c, Tr = 1 - Re, P = .25 + .5 * Re, RP = Re / P, TP = Tr / (1 - P);
        return texture.emission + f.mult(depth > 2 ? (erand48(X) < P ?
                                                      basic_pt(scene, reflray, depth, X) * RP :
                                                      basic_pt(scene, Ray(x, d), depth, X) * TP) :
                                         basic_pt(scene, reflray, depth, X) * Re +
                                         basic_pt(scene, Ray(x, d), depth, X) * Tr);
    }
}