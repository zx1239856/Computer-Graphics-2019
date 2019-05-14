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
    utils::Vector3 x = ray.getVector(std::get<1>(intersect)), n = std::get<3>(intersect), nl =
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
        double a = erand48(X);
        double s = 1;
        if(a < texture.rho_s) {
            // phong specular
            double phi = 2 * M_PI * erand48(X), r2 = erand48(X);
            double cos_theta = pow(1 - r2, 1 / (1 + texture.phong_s));
            double sin_theta = sqrt(1 - cos_theta * cos_theta);
            Vector3 w = ray.direction.reflect(nl), u = ((fabs(w.x()) > .1 ? Vector3(0, 1) : Vector3(1)).cross(nl)).normalize(), v = w.cross(u).normalize();
            Vector3 d = (u * cos(phi) * sin_theta + v * sin(phi) * sin_theta + w * cos_theta).normalize();
            return texture.emission + f.first.mult(basic_pt(scene, Ray(x, d), depth, X));
        }
        else if(a < texture.rho_d) {
            double r1 = 2 * M_PI * erand48(X), r2s = sqrt(erand48(X));
            Vector3 w = nl, u = ((fabs(w.x()) > .1 ? Vector3(0, 1) : Vector3(1)).cross(nl)).normalize(), v = w.cross(u).normalize();
            Vector3 d = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2s * r2s)).normalize();
            return texture.emission + f.first.mult(basic_pt(scene, Ray(x, d), depth, X));
        }
        else
            return texture.emission;
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