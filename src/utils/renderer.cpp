#include "renderer.h"
#include "scene.hpp"
#include "object.hpp"

utils::Vector3 basic_pt(const Scene &scene, const Ray &ray, int depth, unsigned short *X, bool ray_cast) {
    using namespace utils;
    auto intersect = scene.findFirstIntersect(ray);
    if (std::get<0>(intersect) == -1)
        return utils::Vector3(); // no intersection
    BasicObject *obj = scene.object(std::get<0>(intersect));
    if (obj == nullptr)
        return utils::Vector3();
    bool into = false;
    utils::Vector3 x = ray.getVector(std::get<1>(intersect)), n = std::get<3>(intersect), nl =
            n.dot(ray.direction) < 0 ? into = true, n : -n;
    const Texture &texture = obj->texture;
    auto &&f = texture.getColor(std::get<2>(intersect), X);
    double p = f.first.max();
    if (p < EPSILON)
        return ray_cast ? Vector3() : texture.emission;
    if (++depth > PATH_TRACING_MAX_DEPTH) {
        if (erand48(X) < p)f.first = f.first * (1 / p);
        return ray_cast ? Vector3() : texture.emission;
    }
    if (f.second == DIFF) {
        double a = erand48(X);
        if (a < texture.brdf.rho_s) {
            // phong specular
            double phi = 2 * M_PI * erand48(X), r2 = erand48(X);
            double cos_theta = pow(1 - r2, 1 / (1 + texture.brdf.phong_s));
            double sin_theta = sqrt(1 - cos_theta * cos_theta);
            Vector3 w = ray.direction.reflect(nl), u = ((fabs(w.x()) > .1 ? Vector3(0, 1) : Vector3(1)).cross(
                    w)).normalize(), v = w.cross(u).normalize();
            Vector3 d = (u * cos(phi) * sin_theta + v * sin(phi) * sin_theta + w * cos_theta).normalize();
            return (ray_cast ? Vector3() : texture.emission) +
                   f.first.mult(basic_pt(scene, Ray(x, d), depth, X, ray_cast));
        } else if (a < texture.brdf.rho_d) {
            double r1 = 2 * M_PI * erand48(X), r2s = sqrt(erand48(X));
            Vector3 w = nl, u = ((fabs(w.x()) > .1 ? Vector3(0, 1) : Vector3(1)).cross(nl)).normalize(), v = w.cross(
                    u).normalize();
            Vector3 d = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2s * r2s)).normalize();
            if (!ray_cast)
                return (ray_cast ? Vector3() : texture.emission) +
                       f.first.mult(basic_pt(scene, Ray(x, d), depth, X, ray_cast));
            else
                return f.first;
        } else
            return (ray_cast ? Vector3() : texture.emission);
    } else if (f.second == SPEC)
        return (ray_cast ? Vector3() : texture.emission) +
               f.first.mult(basic_pt(scene, Ray(x, ray.direction.reflect(nl)), depth, X, ray_cast));
    else {
        Ray reflray = Ray(x, ray.direction.reflect(nl));
        Vector3 d = ray.direction.refract(n, into ? 1 : texture.brdf.re_idx, into ? texture.brdf.re_idx : 1);
        if (d.len2() < EPSILON) // total internal reflection
            return (ray_cast ? Vector3() : texture.emission) +
                   f.first.mult(basic_pt(scene, reflray, depth, X, ray_cast));
        double a = texture.brdf.re_idx - 1, b = texture.brdf.re_idx + 1, R0 = a * a / (b * b), c =
                1 - (into ? -ray.direction.dot(nl) : d.dot(n));
        double Re = R0 + (1 - R0) * c * c * c * c * c, Tr = 1 - Re, P = .25 + .5 * Re, RP = Re / P, TP = Tr / (1 - P);
        return (ray_cast ? Vector3() : texture.emission) + f.first.mult(depth > 2 ? (erand48(X) < P ?
                                                                                     basic_pt(scene, reflray, depth, X,
                                                                                              ray_cast) * RP :
                                                                                     basic_pt(scene, Ray(x, d), depth,
                                                                                              X, ray_cast) * TP) :
                                                                        basic_pt(scene, reflray, depth, X, ray_cast) *
                                                                        Re +
                                                                        basic_pt(scene, Ray(x, d), depth, X, ray_cast) *
                                                                        Tr);
    }
}


void ray_trace(const Scene &scene, const Ray &ray, const utils::Vector3 &flux, const utils::Vector3 &weight, int depth,
               unsigned short *X,
               HitPointKDTree &tree, const bool &cam_pass, const uint32_t &pixel_place) {
    using namespace utils;
    if (depth > RAY_TRACING_MAX_DEPTH)
        return;
    auto intersect = scene.findFirstIntersect(ray); // <obj_id, t, surface, norm>
    if (std::get<0>(intersect) == -1)
        return;
    BasicObject *obj = scene.object(std::get<0>(intersect));
    if (obj == nullptr)
        return;
    bool into = false;
    utils::Vector3 hit = ray.getVector(std::get<1>(intersect)), n = std::get<3>(intersect), norm =
            n.dot(ray.direction) < 0 ? into = true, n : -n;
    const Texture &texture = obj->texture;
    auto color = texture.getColor(std::get<2>(intersect), X);
    double temp = erand48(X);
    auto new_weight = weight.mult(color.first);
    auto new_flux = flux.mult(color.first);
    auto d_reflect = ray.direction.reflect(norm);
    if (temp <= texture.brdf.specular) {
        ray_trace(scene, Ray(hit + d_reflect * EPSILON, d_reflect), new_flux, new_weight, depth + 1, X, tree, cam_pass,
                  pixel_place);
    } else if (temp <= texture.brdf.diffuse) {
        double ac = erand48(X);
        if (ac <= texture.brdf.rho_s) {
            double phi = 2 * M_PI * erand48(X), r2 = erand48(X);
            double cos_theta = pow(1 - r2, 1 / (1 + texture.brdf.phong_s));
            double sin_theta = sqrt(1 - cos_theta * cos_theta);
            Vector3 w = ray.direction.reflect(norm), u = ((fabs(w.x()) > .1 ? Vector3(0, 1) : Vector3(1)).cross(
                    w)).normalize(), v = w.cross(u).normalize();
            Vector3 d = (u * cos(phi) * sin_theta + v * sin(phi) * sin_theta + w * cos_theta).normalize();
            ray_trace(scene, Ray(hit + d * EPSILON, d), new_flux, new_weight, depth + 1, X, tree, cam_pass,
                      pixel_place);
        } else if (ac <= texture.brdf.rho_d) {
            if (cam_pass) {
                HitPoint &hp = tree.hit_pnts[pixel_place];
                hp.p = hit, hp.weight += new_weight, hp.brdf = &texture.brdf, hp.norm = norm;
                if (texture.emission.len() > EPSILON_3) {
                    hp.valid = false, hp.flux_d += weight.mult(texture.emission);
                } else hp.valid = true;
            } else {
                // photon pass
                tree.update(tree.getRoot(), hit, flux / M_PI, ray.direction);
                double r1 = 2 * M_PI * erand48(X), r2s = sqrt(erand48(X));
                Vector3 w = norm, u = ((fabs(w.x()) > .1 ? Vector3(0, 1) : Vector3(1)).cross(
                        norm)).normalize(), v = w.cross(
                        u).normalize();
                Vector3 d = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2s * r2s)).normalize();
                auto p = color.first.max();
                if (erand48(X) < p)
                    ray_trace(scene, Ray(hit + d * EPSILON, d), new_flux / p, weight, depth + 1, X, tree, cam_pass,
                              pixel_place);
            }
        }
    } else if (temp <= texture.brdf.refraction) {
        Ray reflray(hit + d_reflect * EPSILON, d_reflect);
        Vector3 d = ray.direction.refract(n, into ? 1 : texture.brdf.re_idx, into ? texture.brdf.re_idx : 1);
        if (d.len2() < EPSILON) // total internal reflection
        {
            ray_trace(scene, reflray, flux, weight, depth + 1, X, tree, cam_pass, pixel_place);
        } else {
            double a = texture.brdf.re_idx - 1, b = texture.brdf.re_idx + 1, R0 = a * a / (b * b), c =
                    1 - (into ? -ray.direction.dot(norm) : d.dot(n));
            double Re = R0 + (1 - R0) * c * c * c * c * c;
            if (cam_pass) {
                ray_trace(scene, reflray, flux, new_weight * Re, depth + 1, X, tree, cam_pass, pixel_place);
                ray_trace(scene, Ray(hit + d * EPSILON, d), flux, new_weight * (1 - Re), depth + 1, X, tree, cam_pass,
                          pixel_place);
            } else {
                (erand48(X) <= Re) ?
                ray_trace(scene, reflray, flux, new_weight, depth + 1, X, tree, cam_pass, pixel_place) :
                ray_trace(scene, Ray(hit + d * EPSILON, d), flux, new_weight, depth + 1, X, tree, cam_pass,
                          pixel_place);
            }
        }
    }
}

void evalRadiance(std::vector<utils::Vector3> &out, const utils::Vector3 &light, const int w, const int h,
                  const HitPointKDTree &tree,
                  const int num_rounds, const int num_photons, const int super_sampling) {
    for (int v = 0; v < h; ++v) {
        for (int u = 0; u < w; ++u) {
            const auto &hp = tree.hit_pnts[v * w + u];
            if (num_photons == 0)
                out[(h - v - 1) * w + u] = hp.weight;
            else
                out[(h - v - 1) * w + u] =
                        hp.flux_i / (M_PI * hp.r_sqr * num_rounds * num_photons) + light.mult(hp.flux_d) / num_rounds;
        }
    }
}