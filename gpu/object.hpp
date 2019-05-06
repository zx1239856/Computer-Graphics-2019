#pragma once
/*
 * Objects changed for GPU code
 * Safe for bit-wise copy if mapped texture is not present
 */
#include "cuda_helpers.h"
#include "math_helpers.h"
#include "../common/simple_intersect_algs.hpp"
#include "../utils/bezier.hpp"


struct Texture_GPU {
    utils::Vector3 color, emission;
    Refl_t refl_1, refl_2;
    double probability; // probability of second REFL type
    double re_idx;
    cudaTextureObject_t mapped_image;
    int img_w = 0, img_h = 0;
    utils::Transform2D mapped_transform;
    ~Texture_GPU() {
        CUDA_SAFE_CALL(cudaDestroyTextureObject(mapped_image));
    }

    __device__ pair<utils::Vector3, Refl_t>

    getColor(const utils::Point2D &surface_coord, curandState *state) const {
        if (!img_h || !img_w) // no texture mapping. use default color
        {
            if (curand_uniform_double(state) < probability)
                return {color, refl_2};
            else
                return {color, refl_1};
        } else {
            const int &w = img_w, &h = img_h;
            utils::Point2D p = mapped_transform.transform(surface_coord);
            int u = (lround(w * p.y + .5) % w + w) % w, v =
                    (lround(h * p.x + .5) % h + h) % h;
            uchar4 cc = tex1Dfetch<uchar4>(mapped_image, v * w + u);
            auto color = utils::Vector3(cc.x / 255. * 0.999, cc.y / 255. * 0.999, cc.z / 255. * 0.999); // 8 bit per channel, so color is in [0, 255]
            if (curand_uniform_double(state) < probability)
                return {color, refl_2};
            else if ((cc.x >= 235 || cc.y >= 235 || cc.z >= 235) && curand_uniform_double(state) < 0.13)
                return {color, SPEC};
            else
                return {color, refl_1};
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
               Refl_t _refl, double _re_idx) : origin(o), radius(r) {
        texture.color = _color, texture.emission = _emission, texture.refl_1 = _refl,
        texture.re_idx = _re_idx, texture.probability = 0;
    }

    __device__ triplet<utils::Vector3, double, utils::Point2D>

    intersect(const Ray &ray) const {
        return intersectSphereObject(ray, origin, radius);
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
              Refl_t refl, double re_idx) : n(norm.normalize()), d(dis) {
        texture.color = color, texture.emission = emission, texture.refl_1 = refl,
        texture.re_idx = re_idx, texture.probability = 0;
        preparePlaneObject(n, d, xp, yp, origin);
    }

    __device__ triplet<utils::Vector3, double, utils::Point2D>

    intersect(const Ray &ray) const {
        return intersectPlaneObject(ray, n, d, origin, xp, yp);
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
             Refl_t refl, double re_idx) : p0(min(_p0, _p1)), p1(max(_p0, _p1)) {
        texture.color = color, texture.emission = emission, texture.refl_1 = refl,
        texture.re_idx = re_idx, texture.probability = 0;
    }

    __device__ triplet<utils::Vector3, double, utils::Point2D>

    intersect(const Ray &ray) const {
        double t = intersectAABB(ray, p0, p1);
        return {ray.getVector(t), t, utils::Point2D(0, 0)}; // surface coordinate not available for cube
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
                     Refl_t refl, double re_idx) : axis(_axis), bezier(_bezier) {
        texture.color = color, texture.emission = emission, texture.refl_1 = refl,
        texture.re_idx = re_idx, texture.probability = 0;
    }

    __device__ triplet<utils::Vector3, double, utils::Point2D>

    intersect(const Ray &ray) const {
        auto aabb = boundingBox();
        return intersectBezierObject(ray, axis, bezier, aabb.first, aabb.second);
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
