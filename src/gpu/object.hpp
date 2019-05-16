#pragma once
/*
 * Objects changed for GPU code
 * Safe for bit-wise copy if mapped texture is not present
 */
#include "cuda_helpers.h"
#include "math_helpers.h"
#include "../common/simple_intersect_algs.hpp"
#include "../utils/bezier.hpp"
#include "../utils/kdtree.hpp"


struct Texture_GPU {
    utils::Vector3 color, emission;
    double specular, diffuse, refraction;
    double rho_d, rho_s, phong_s;
    double re_idx;
    cudaTextureObject_t mapped_image;
    int img_w = 0, img_h = 0;
    utils::Transform2D mapped_transform;

    ~Texture_GPU() {
        CUDA_SAFE_CALL(cudaDestroyTextureObject(mapped_image));
    }

    __host__ __device__ void setBRDF(const BRDF &brdf) {
        specular = brdf.specular, diffuse = brdf.diffuse, refraction = brdf.refraction,
        rho_d = brdf.rho_d, rho_s = brdf.rho_s, phong_s = brdf.phong_s, re_idx = brdf.re_idx;
        // normalization
        double s = specular + diffuse + refraction;
        specular /= s, diffuse /= s, refraction /= s;
        diffuse += specular;
        refraction += diffuse;
        s = rho_s + rho_d;
        rho_s /= s, rho_d /= s;
        rho_d += rho_s;
    }

    __device__ pair<utils::Vector3, Refl_t>

    getColor(const utils::Point2D &surface_coord, curandState *state) const {
        double prob = curand_uniform_double(state);
        if (!img_h || !img_w) // no texture mapping. use default color
        {
            if (prob < specular)
                return {color, SPEC};
            else if (prob < diffuse)
                return {color, DIFF};
            else
                return {color, REFR};
        } else {
            const int &w = img_w, &h = img_h;
            utils::Point2D p = mapped_transform.transform(surface_coord);
            int u = (lround(w * p.y + .5) % w + w) % w, v =
                    (lround(h * p.x + .5) % h + h) % h;
            uchar4 cc = tex1Dfetch<uchar4>(mapped_image, v * w + u);
            auto color = utils::Vector3(cc.x / 255. * 0.999, cc.y / 255. * 0.999,
                                        cc.z / 255. * 0.999); // 8 bit per channel, so color is in [0, 255]
            if (prob < specular)
                return {color, SPEC};
            else if (prob < diffuse) {
                if ((cc.x >= 235 || cc.y >= 235 || cc.z >= 235) && curand_uniform_double(state) < 0.13)
                    return {color, SPEC};
                else return {color, DIFF};
            } else
                return {color, REFR};
        }
    }
};

struct Sphere_GPU {
    utils::Vector3 origin;
    double radius;
    Texture_GPU texture;

    __host__ __device__

    Sphere_GPU(const utils::Vector3 &o, double r, const Texture_GPU &t) : origin(o), radius(r), texture(t) {}

    __host__ __device__

    Sphere_GPU(const utils::Vector3 &o, double r, const utils::Vector3 &_color, const utils::Vector3 &_emission,
               const BRDF &brdf) : origin(o), radius(r) {
        texture.color = _color, texture.emission = _emission;
        texture.setBRDF(brdf);
    }

    __device__ triplet<utils::Vector3, double, utils::Point2D>

    intersect(const Ray &ray) const {
        auto res = intersectSphereObject(ray, origin, radius);
        res.first = norm(res.first);
        return res;
    }

    __device__ pair<utils::Vector3, utils::Vector3>

    boundingBox() const {
        return {origin - radius, origin + radius};
    }

    __device__ utils::Vector3

    norm(const utils::Vector3 &vec, const utils::Point2D &unused = utils::Point2D(0, 0)) const {
        return (vec - origin).normalize();
    }
};

struct Plane_GPU {
    utils::Vector3 n;
    double d;
    utils::Vector3 xp, yp;
    utils::Vector3 origin;
    Texture_GPU texture;

    __host__ __device__

    Plane_GPU(const utils::Vector3 &norm, double dis, const Texture_GPU &t) : n(norm.normalize()), d(dis), texture(t) {
        preparePlaneObject(n, d, xp, yp, origin);
    }

    __host__ __device__

    Plane_GPU(const utils::Vector3 &norm, double dis, const utils::Vector3 &color, const utils::Vector3 &emission,
              const BRDF &brdf) : n(norm.normalize()), d(dis) {
        texture.color = color, texture.emission = emission;
        texture.setBRDF(brdf);
        preparePlaneObject(n, d, xp, yp, origin);
    }

    __device__ triplet<utils::Vector3, double, utils::Point2D>

    intersect(const Ray &ray) const {
        auto res = intersectPlaneObject(ray, n, d, origin, xp, yp);
        res.first = n;
        return res;
    }

    __device__ pair<utils::Vector3, utils::Vector3>

    boundingBox() const {
        return {n, n}; // the bounding box of a plane is ill-defined
    }

    __device__ utils::Vector3

    norm(const utils::Vector3 &vec = utils::Vector3(),
         const utils::Point2D &unused = utils::Point2D(0, 0)) const {
        return n;
    }
};

struct Cube_GPU {
    utils::Vector3 p0, p1;
    Texture_GPU texture;
    __host__ __device__

    Cube_GPU(const utils::Vector3 &_p0, const utils::Vector3 &_p1, const Texture_GPU &t) : p0(min(_p0, _p1)),
                                                                                           p1(max(_p0, _p1)),
                                                                                           texture(t) {}

    __host__ __device__

    Cube_GPU(const utils::Vector3 &_p0, const utils::Vector3 &_p1, const utils::Vector3 &color,
             const utils::Vector3 &emission,
             const BRDF &brdf) : p0(min(_p0, _p1)), p1(max(_p0, _p1)) {
        texture.color = color, texture.emission = emission;
        texture.setBRDF(brdf);
    }

    __device__ triplet<utils::Vector3, double, utils::Point2D>

    intersect(const Ray &ray) const {
        double t = intersectAABB(ray, p0, p1);
        return {norm(ray.getVector(t)), t, utils::Point2D(0, 0)}; // surface coordinate not available for cube
    }

    __device__ pair<utils::Vector3, utils::Vector3>

    boundingBox() const {
        return {p0, p1};
    }

    __device__ utils::Vector3

    norm(const utils::Vector3 &vec, const utils::Point2D &unused = utils::Point2D(0, 0)) const {
        return normOfAABB(vec, p0, p1);
    }
};

struct RotaryBezier_GPU {
    utils::Vector3 axis;
    utils::Bezier2D_GPU bezier;
    Texture_GPU texture;

    // rotate bezier2d curve along y-axis
    __host__ __device__

    RotaryBezier_GPU(const utils::Vector3 &_axis, const utils::Bezier2D_GPU &_bezier, const Texture_GPU &t) :
            axis(_axis), bezier(_bezier), texture(t) {}

    __host__ __device__

    RotaryBezier_GPU(const utils::Vector3 &_axis, const utils::Bezier2D_GPU &_bezier, const utils::Vector3 &color,
                     const utils::Vector3 &emission,
                     const BRDF &brdf) : axis(_axis), bezier(_bezier) {
        texture.color = color, texture.emission = emission;
        texture.setBRDF(brdf);
    }

    __device__ triplet<utils::Vector3, double, utils::Point2D>

    intersect(const Ray &ray) const {
        auto aabb = boundingBox();
        auto res = intersectBezierObject(ray, axis, bezier, aabb.first, aabb.second);
        return {norm(res.first, res.third), res.second, res.third};
    }

    __device__ pair<utils::Vector3, utils::Vector3>

    boundingBox() const {
        return {utils::Vector3(-bezier.xmax + axis.x() - 0.5, bezier.ymin + axis.y() - 0.5,
                               -bezier.xmax + axis.z() - 0.5),
                utils::Vector3(bezier.xmax + axis.x() + 0.5, bezier.ymax + axis.y() + 0.5,
                               bezier.xmax + axis.z() + 0.5)};
    }

    __device__ utils::Vector3

    norm(const utils::Vector3 &vec, const utils::Point2D &surface_coord) const {
        return normOfBezier(vec, surface_coord, bezier);
    }
};

struct TriangleMeshObject_GPU {
    utils::Vector3 pos;
    double ratio;
    utils::KDTree_GPU kd_tree;
    Texture_GPU texture;

    __host__ __device__

    TriangleMeshObject_GPU(const utils::Vector3 &_pos, double _ratio, const utils::KDTree_GPU &gpu_tree,
                           const Texture_GPU &t) :
            pos(_pos), ratio(_ratio), kd_tree(gpu_tree), texture(t) {}


    __host__ __device__

    TriangleMeshObject_GPU(const utils::Vector3 &_pos, double _ratio, const utils::KDTree_GPU &gpu_tree, const utils::Vector3 &color,
                     const utils::Vector3 &emission,
                     const BRDF &brdf) : pos(_pos), ratio(_ratio), kd_tree(gpu_tree) {
        texture.color = color, texture.emission = emission;
        texture.setBRDF(brdf);
    }

    __device__ triplet<utils::Vector3, double, utils::Point2D>

    intersect(const Ray &ray) const {
        auto r = ray;
        r.origin = (r.origin - pos) / ratio;
        auto res = kd_tree.singleRayStacklessIntersect(r);
        if (res.first >= INF)
            return {utils::Vector3(), INF, utils::Point2D()};
        auto hit_in_world = r.getVector(res.first) * ratio + pos;
        res.first = (abs(r.direction.x()) > abs(r.direction.y()) &&
                     abs(r.direction.x()) > abs(r.direction.z())) ?
                    (hit_in_world - ray.origin).x() / r.direction.x() : ((abs(r.direction.y()) >
                                                                          abs(r.direction.z())) ?
                                                                         (hit_in_world - ray.origin).y() /
                                                                         r.direction.y() :
                                                                         (hit_in_world - ray.origin).z() /
                                                                         r.direction.z());
        return {res.second, res.first, utils::Point2D()};
    }

    __device__ pair<utils::Vector3, utils::Vector3>

    boundingBox() const {
        auto res = kd_tree.getAABB();
        res.first = res.first * ratio + pos - 0.5;
        res.second = res.second * ratio + pos + 0.5;
        return res;
    }

    __device__ utils::Vector3

    norm(const utils::Vector3 &vec, const utils::Point2D &surface_coord) const {
        return utils::Vector3();  // dummy norm
    }
};
