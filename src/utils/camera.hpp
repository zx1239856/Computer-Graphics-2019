#pragma once

#include "../common/geometry.hpp"
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

    std::vector<utils::Vector3> renderPPM(int num_rounds, int num_photons) {
        std::vector<utils::Vector3> c(w * h);
        if (scene == nullptr)
            return c;
        utils::Vector3 cx = utils::Vector3(-direction.z(), 0, direction.x());
        if (cx.len2() < EPSILON) {
            cx = utils::Vector3(1, 0, 0) * w / h * fov;  // camera pointing towards y-axis
        } else cx = cx.normalize() * w / h * fov;
        utils::Vector3 cy = cx.cross(direction).normalize() * fov, r;
        // cx = direction \times \cap{y}, cy = cx \times direction

        HitPointKDTree tree(w * h);

        for (int rounds = 0; rounds < num_rounds; ++rounds) {
            // ray-tracing pass
            fprintf(stderr, "SPPM Pass #%d\n", rounds);
            tree.clearPreviousWeight();
#pragma omp parallel for schedule(dynamic, 1)
            for (int v = 0; v < h; ++v) {
                fprintf(stderr, "\rRay tracing %5.2lf%%", v * 100. / h);
                for (int u = 0; u < w; ++u) {
                    //utils::Vector3 avg_d;
                    /*for (int sy = 0; sy < 2; ++sy) {  // 2x2 super sampling
                        for (int sx = 0; sx < 2; ++sx) {
                            ushort X[3] = {static_cast<ushort>(v + sx), static_cast<ushort>(v * u + sy),
                                           static_cast<ushort>(v * u * v + sx * sy)};
                            double r1 = 2 * erand48(X), dx = r1 < 1 ? sqrt(r1) : 2 - sqrt(2 - r1);
                            double r2 = 2 * erand48(X), dy = r2 < 1 ? sqrt(r2) : 2 - sqrt(2 - r2);
                            utils::Vector3 d = cx * (((sx + .5 + dx) / 2 + u) / w - .5) +
                                               cy * (((sy + .5 + dy) / 2 + v) / h - .5) + direction;
                            double cos = d.normalize().dot(direction.normalize());
                            utils::Vector3 hit = origin + d * focal_dist / cos;  // real hit point on focal plane
                            utils::Vector3 p_origin = origin +
                                                      (utils::Vector3(erand48(X) * 1.01, erand48(X), erand48(X)) - .5) *
                                                      2 *
                                                      aperture; // origin perturbation
                            d = d.normalize();
                            avg_d += d;
                            ray_trace(*scene, Ray(p_origin, (hit - p_origin).normalize()), utils::Vector3(1, 1, 1), 0,
                                      X, tree, true, v * w + u);
                        }
                    }
                    tree.hit_pnts[v * w + u].d = avg_d * -.25;*/
		    int sx = rounds & 1, sy = rounds & 2;
                    ushort X[3] = {static_cast<ushort>(v * u + sx), static_cast<ushort>(v * u + w + time(nullptr) + sy),
                                   static_cast<ushort>(v * u * v * w + time(nullptr))};
                    double r1 = 2 * erand48(X), dx = r1 < 1 ? sqrt(r1) : 2 - sqrt(2 - r1);
                    double r2 = 2 * erand48(X), dy = r2 < 1 ? sqrt(r2) : 2 - sqrt(2 - r2);
                    utils::Vector3 d = cx * (((sx + .5 + dx) / 2 + u) / w - .5) +
                                               cy * (((sy + .5 + dy) / 2 + v) / h - .5) + direction;
                    double cos = d.normalize().dot(direction.normalize());
                    utils::Vector3 hit = origin + d * focal_dist / cos;  // real hit point on focal plane
                    double theta = erand48(X) * M_PI * 2;
                    utils::Vector3 p_origin = origin +
                                              (cx * std::cos(theta) + cy * std::sin(theta)) * erand48(X) *
                                              2 *
                                              aperture; // origin perturbation
                    tree.hit_pnts[v * w + u].d = -d;
                    ray_trace(*scene, Ray(p_origin, (hit - p_origin).normalize()), utils::Vector3(),
                              utils::Vector3(1, 1, 1), 0,
                              X, tree, true, v * w + u);
                }
            }
            fprintf(stderr, "\rRay tracing **done**");
            if (!rounds)
                tree.setInitialRadiusHeuristic(w, h);
            tree.initializeTree();

            // photon-tracing pass
#pragma omp parallel for schedule(dynamic, 1)
            for (int i = 0; i < num_photons; ++i) {
                ushort X[3] = {static_cast<ushort>(num_photons + i), static_cast<ushort>(w * h + i + num_rounds),
                               static_cast<ushort>(w * h + i + time(nullptr))};
                fprintf(stderr, "\rPhoton tracing %5.2lf%%", i * 100. / num_photons);
                Ray ray = scene->genRay(X);
                ray_trace(*scene, ray, utils::Vector3(40000, 40000, 40000) * (M_PI * 4.0), utils::Vector3(3., 3., 3.), 0,
                          X, tree, false, 0);
            }
            fprintf(stderr, "\rPhoton tracing **done**\n");

            tree.updateHitPointStats();

            int ccc = 0, ccc2 = 0;
            unsigned long long collected_photons = 0;
            double r_min = INF, r_max = 0;
            for (int i = 0; i < w * h; ++i) {
                collected_photons += tree.hit_pnts[i].n;
                r_min = std::min(r_min, tree.hit_pnts[i].r_sqr);
                r_max = std::max(r_max, tree.hit_pnts[i].r_sqr);
            }
            fprintf(stderr, "Collected Photons: %llu\nSearch radius: min: %lf, max: %lf\n", collected_photons, r_min,
                    r_max);

            for (int i = 0; i < w * h; ++i) {
                const auto &hp = tree.hit_pnts[i];
                if (hp.flux_i.len() > EPSILON_2)
                    ++ccc;
                if (hp.flux_d.len() > EPSILON_2)
                    ++ccc2;
            }
            fprintf(stderr, "Hit Point with indirect flux: %d\t direct flux:%d\n\n", ccc, ccc2);
        }
        // eval radiance
        evalRadiance(c, utils::Vector3(1, 1, 1), w, h, tree, num_rounds, num_photons);
        //evalRadianceTest(c, w, h, tree);  // this is for ray cast test
        return c;
    }

    std::vector<utils::Vector3> renderPt(int samps, bool ray_cast = false) {
        std::vector<utils::Vector3> c(w * h);
        if (scene == nullptr)
            return c;
        utils::Vector3 cx = utils::Vector3(-direction.z(), 0, direction.x());
        if (cx.len2() < EPSILON) {
            cx = utils::Vector3(1, 0, 0) * w / h * fov;  // camera pointing towards y-axis
        } else cx = cx.normalize() * w / h * fov;
        utils::Vector3 cy = cx.cross(direction).normalize() * fov, r;
        // cx = direction \times \cap{y}, cy = cx \times direction
#pragma omp parallel for schedule(dynamic, 1) private(r)
        for (int v = 0; v < h; ++v) {
            fprintf(stderr, "\rRendering (%d spp) %5.2f%%", samps * 4, 100. * v / (h - 1));
            for (int u = 0; u < w; ++u) {
                for (int sy = 0; sy < 2; ++sy) {  // 2x2 super sampling
                    for (int sx = 0; sx < 2; ++sx) {
                        ushort X[3] = {static_cast<ushort>(v + sx), static_cast<ushort>(v * u + sy),
                                       static_cast<ushort>(v * u * v + sx * sy)};
                        r = utils::Vector3();
                        for (int s = 0; s < samps; ++s) {
                            double r1 = 2 * erand48(X), dx = r1 < 1 ? sqrt(r1) : 2 - sqrt(2 - r1);
                            double r2 = 2 * erand48(X), dy = r2 < 1 ? sqrt(r2) : 2 - sqrt(2 - r2);
                            utils::Vector3 d = cx * (((sx + .5 + dx) / 2 + u) / w - .5) +
                                               cy * (((sy + .5 + dy) / 2 + v) / h - .5) + direction;
                            double cos = d.normalize().dot(direction.normalize());
                            utils::Vector3 hit = origin + d * focal_dist / cos;  // real hit point on focal plane
                            double theta = erand48(X) * M_PI * 2;
                    utils::Vector3 p_origin = origin +
                                              (cx * std::cos(theta) + cy * std::sin(theta)) * erand48(X) *
                                              2 *
                                              aperture; // origin perturbation
                            r = r + basic_pt(*scene, Ray(p_origin, (hit - p_origin).normalize()), 0, X, ray_cast) * (1. / samps);
                        }
                        c[(h - v - 1) * w + u] += r.clamp(0, 1) * .25;
                    }
                }
            }
        }
        return c;
    }
};
