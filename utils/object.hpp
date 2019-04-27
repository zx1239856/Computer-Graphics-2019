#pragma once

#include <vector>
#include <functional>
#include <cassert>
#include <cmath>
#include "../common/common.h"
#include "../common/geometry.hpp"
#include "../common/ray.hpp"
#include "bezier.hpp"

/*
 * Common intersections(sphere, axis-aligned cubes)
 */

inline double intersectSphere(const Ray &ray, const utils::Vector3 &o, double r) {
    utils::Vector3 op = o - ray.origin;
    double b = op.dot(ray.direction);
    double d = b * b - op.len2() + r * r;
    if (d < 0)
        return INF;
    else {
        d = std::sqrt(d);
        double t = (b - d > EPSILON) ? b - d : ((b + d) > EPSILON ? b + d : -1);
        if (t < 0)
            return INF;
        return t;
    }
}

inline double intersectAABB(const Ray &ray, const utils::Vector3 &p0, const utils::Vector3 &p1) {
    double t, t_min = INF;
    auto check_tmin = [&t, &t_min](std::function<bool()> const &inrange) {
        if (t > 0 && t < t_min)
            if (inrange())
                t_min = t;
    };
    auto xcheck = [&]() -> bool {
        auto v = ray.getVector(t);
        return p0.y() <= v.y() && v.y() <= p1.y() && p0.z() <= v.z() && v.z() <= p1.z();
    };
    auto ycheck = [&]() -> bool {
        auto v = ray.getVector(t);
        return p0.x() <= v.x() && v.x() <= p1.x() && p0.z() <= v.z() && v.z() <= p1.z();
    };
    auto zcheck = [&]() -> bool {
        auto v = ray.getVector(t);
        return p0.y() <= v.y() && v.y() <= p1.y() && p0.x() <= v.x() && v.x() <= p1.x();
    };
    if (ray.direction.x() > 0) // p0 is visible
    {
        t = (p0.x() - ray.origin.x()) / ray.direction.x();
        check_tmin(xcheck);
    } else {
        t = (p1.x() - ray.origin.x()) / ray.direction.x();
        check_tmin(xcheck);
    }
    if (ray.direction.y() > 0) {
        t = (p0.y() - ray.origin.y()) / ray.direction.y();
        check_tmin(ycheck);
    } else {
        t = (p1.y() - ray.origin.y()) / ray.direction.y();
        check_tmin(ycheck);
    }
    if (ray.direction.z() > 0) {
        t = (p0.z() - ray.origin.z()) / ray.direction.z();
        check_tmin(zcheck);
    } else {
        t = (p1.z() - ray.origin.z()) / ray.direction.z();
        check_tmin(zcheck);
    }
    if (t_min < INF)
        return t_min;
    else
        return INF;
}

struct TexturePT {
    utils::Vector3 color, emission;
    Refl_t refl_1, refl_2;
    double probability; // probability of second REFL type
    double re_idx;
    std::vector<std::vector<utils::Vector3>> mapped_image;
    utils::Transform2D mapped_transform;

    TexturePT(const utils::Vector3 &_color, const utils::Vector3 &_emission, Refl_t _refl, double _re_idx) :
            color(_color), emission(_emission), refl_1(_refl), refl_2(DIFF), probability(0), re_idx(_re_idx) {}

    TexturePT(const utils::Vector3 &_color, const utils::Vector3 &_emission, Refl_t _refl_1, Refl_t _refl_2,
              double _prob, double _re_idx) :
            color(_color), emission(_emission), refl_1(_refl_1), refl_2(_refl_2), probability(_prob), re_idx(_re_idx) {}

    std::pair<utils::Vector3, Refl_t> getColor(const utils::Point2D &surface_coord, unsigned short X[3]) const {
        if (!mapped_image.size()) // no texture mapping. use default color
        {
            if (erand48(X) < probability)
                return {color, refl_2};
            else
                return {color, refl_1};
        } else {
            const int &w = mapped_image[0].size(), &h = mapped_image.size();
            utils::Point2D p = mapped_transform.transform(surface_coord);
            int u = (lround(w * p.y + .5) % w + w) % w, v =
                    (lround(h * p.x + .5) % h + h) % h;
            auto color = mapped_image[v][u]; // 8 bit per channel, so color is in [0, 255]
            if (erand48(X) < probability)
                return {color / 255. * 0.999, refl_2};
            else if (color.x() >= 235 || color.y() >= 235 || color.z() >= 235)
                return {color / 255. * 0.999, SPEC};
            else
                return {color / 255. * 0.999, refl_1};
        }
    }
};

struct TexturePPM {
    double placeholder; // TODO
};

struct Texture {
    TexturePT pt;
    TexturePPM ppm;

    Texture(const utils::Vector3 &_color, const utils::Vector3 &_emission, Refl_t _refl, double _re_idx) :
            pt(_color, _emission, _refl, _re_idx) {}

    Texture(const utils::Vector3 &_color, const utils::Vector3 &_emission, Refl_t _refl_1, Refl_t _refl_2, double prob,
            double _re_idx) :
            pt(_color, _emission, _refl_1, _refl_2, prob, _re_idx) {}

    Texture(const TexturePT &_pt) : pt(_pt) {}
};

class BasicObject {
public:
    Texture texture;

    BasicObject(const utils::Vector3 &color, const utils::Vector3 &emission, Refl_t refl, double re_idx) :
            texture(color, emission, refl, re_idx) {}

    BasicObject(const Texture &t) : texture(t) {}

    virtual std::pair<utils::Vector3, utils::Vector3> boundingBox() const = 0;

    virtual std::tuple<utils::Vector3, double, utils::Point2D>
    intersect(const Ray &ray) const = 0; // hit_point, distance, surface_coord

    virtual utils::Vector3 norm(const utils::Vector3 &vec, const utils::Point2D &surface_coord) const = 0;

    virtual ~BasicObject() {}
};

class Sphere : public BasicObject {
    utils::Vector3 origin;
    double radius;
public:
    Sphere(const utils::Vector3 &o, double r, const Texture &t) : BasicObject(t), origin(o), radius(r) {}

    Sphere(const utils::Vector3 &o, double r, const utils::Vector3 &_color, const utils::Vector3 &_emission,
           Refl_t _refl, double _re_idx) :
            BasicObject(_color, _emission, _refl, _re_idx), origin(o), radius(r) {}

    virtual std::tuple<utils::Vector3, double, utils::Point2D> intersect(const Ray &ray) const override {
        double t = intersectSphere(ray, origin, radius);
        if (t > 0 && t < INF) {
            auto hit = ray.getVector(t);
            double phi = std::atan2(hit.z() - origin.z(), hit.x() - origin.x());
            return {hit, t, utils::Point2D(std::acos((hit.y() - origin.y()) / radius),
                                           phi < 0 ? phi + PI_DOUBLED
                                                   : phi)}; // spherical coordinate (theta, phi), note y is the rotation axis here
        } else
            return {utils::Vector3(), INF, utils::Point2D(0, 0)};
    }

    virtual std::pair<utils::Vector3, utils::Vector3> boundingBox() const override {
        return {origin - radius, origin + radius};
    }

    virtual utils::Vector3
    norm(const utils::Vector3 &vec, const utils::Point2D &unused = utils::Point2D(0, 0)) const override {
        assert(std::abs((vec - origin).len() - radius) < EPSILON);
        return (vec - origin).normalize();
    }
};

class Plane : public BasicObject {
    utils::Vector3 n;
    double d;
    utils::Vector3 xp, yp;
    utils::Vector3 origin;

    void prepare() {
        if (std::abs(std::abs(n.y()) - 1) < EPSILON_2)
            xp = utils::Vector3(1, 0, 0), yp = utils::Vector3(0, 0, 1);
        else
            xp = n.cross(utils::Vector3(0, 1, 0)).normalize(), yp = xp.cross(n).normalize();
        origin = n * d;
    }

public:
    Plane(const utils::Vector3 &norm, double dis, const Texture &t) : BasicObject(t), n(norm.normalize()), d(dis) {
        prepare();
    }

    Plane(const utils::Vector3 &norm, double dis, const utils::Vector3 &color, const utils::Vector3 &emission,
          Refl_t refl, double re_idx) :
            BasicObject(color, emission, refl, re_idx), n(norm.normalize()), d(dis) {
        prepare();
    }

    std::tuple<utils::Vector3, double, utils::Point2D> intersect(const Ray &ray) const override {
        double prod = ray.direction.dot(n);
        double t = (d - n.dot(ray.origin)) / prod;
        if (t < EPSILON)
            return {utils::Vector3(), INF, utils::Point2D(0, 0)};
        auto &&hit = ray.getVector(t);
        auto vec = hit - origin;
        return {hit, t, utils::Point2D(vec.dot(xp), vec.dot(yp))};  // project the vector to plane
    }

    virtual std::pair<utils::Vector3, utils::Vector3> boundingBox() const override {
        return {n, n}; // the bounding box of a plane is ill-defined
    }

    virtual utils::Vector3
    norm(const utils::Vector3 &vec = utils::Vector3(),
         const utils::Point2D &unused = utils::Point2D(0, 0)) const override {
        return n;
    }
};

class Cube : public BasicObject {
    utils::Vector3 p0, p1;
public:
    Cube(const utils::Vector3 &_p0, const utils::Vector3 &_p1, const Texture &t) : BasicObject(t), p0(min(_p0, _p1)),
                                                                                   p1(max(_p0, _p1)) {}

    Cube(const utils::Vector3 &_p0, const utils::Vector3 &_p1, const utils::Vector3 &color,
         const utils::Vector3 &emission,
         Refl_t refl, double re_idx) :
            BasicObject(color, emission, refl, re_idx), p0(min(_p0, _p1)), p1(max(_p0, _p1)) {}

    virtual std::tuple<utils::Vector3, double, utils::Point2D> intersect(const Ray &ray) const override {
        double t = intersectAABB(ray, p0, p1);
        return {ray.getVector(t), t, utils::Point2D(0, 0)}; // surface coordinate not available for cube
    }

    virtual std::pair<utils::Vector3, utils::Vector3> boundingBox() const override {
        return {p0, p1};
    }

    virtual utils::Vector3
    norm(const utils::Vector3 &vec, const utils::Point2D &unused = utils::Point2D(0, 0)) const override {
        if (abs(vec.x() - p0.x()) < EPSILON || abs(vec.x() - p1.x()) < EPSILON)
            return utils::Vector3(abs(vec.x() - p0.x()) < EPSILON ? -1 : 1, 0, 0);
        if (abs(vec.y() - p0.y()) < EPSILON || abs(vec.y() - p1.y()) < EPSILON)
            return utils::Vector3(0, abs(vec.y() - p0.y()) < EPSILON ? -1 : 1, 0);
        if (abs(vec.z() - p0.z()) < EPSILON || abs(vec.z() - p1.z()) < EPSILON)
            return utils::Vector3(0, 0, abs(vec.z() - p0.z()) < EPSILON ? -1 : 1);
        return utils::Vector3();
    }
};

class RotaryBezier : public BasicObject {
    utils::Vector3 axis;
    utils::Bezier2D bezier;

    // rotate bezier2d curve along y-axis
    double horiz_ray_solver(double y, size_t iter = 15) const {
        double t = .5, yt, dyt;
        for (size_t i = iter; i; --i) {
            if (t < 0) t = EPSILON_2;
            if (t > 1) t = 1 - EPSILON_2;
            yt = bezier.getPoint(t).y - y;
            dyt = bezier.getDerivative(t).y;
            if (std::abs(yt) < EPSILON_1)
                return t;
            t -= yt / dyt;
        }
        return -1;
    }

    double normal_ray_solver(double A, double B, double C, double D, double initial_val, size_t iter = 15) const {
        double t = initial_val, gt, gyt;
        for (size_t i = iter; i; --i) {
            if (t < 0) t = EPSILON_2;
            if (t > 1) t = 1 - EPSILON_2;
            auto xy = bezier.getPoint(t);
            auto dxy = bezier.getDerivative(t);
            double val = (A * xy.y + B);
            gt = val * val - D * xy.x * xy.x + C;
            gyt = 2 * val * A * dxy.y - 2 * D * xy.x * dxy.x;
            if (std::abs(gt) < EPSILON_1)
                return t;
            t -= gt / gyt;
        }
        return -1;
    }

public:
    RotaryBezier(const utils::Vector3 &_axis, const utils::Bezier2D &_bezier, const Texture &t) :
            BasicObject(t), axis(_axis), bezier(_bezier) {}

    RotaryBezier(const utils::Vector3 &_axis, const utils::Bezier2D &_bezier, const utils::Vector3 &color,
                 const utils::Vector3 &emission,
                 Refl_t refl, double re_idx) :
            BasicObject(color, emission, refl, re_idx), axis(_axis), bezier(_bezier) {}

    virtual std::tuple<utils::Vector3, double, utils::Point2D> intersect(const Ray &ray) const override {
        if (std::abs(ray.direction.y()) < EPSILON_3) { // light parallel to x-z plane
            double temp = utils::Vector3(axis.x() - ray.origin.x(), 0, axis.z() - ray.origin.z()).len();
            double initial_y = ray.getVector(temp).y();
            double t_ = horiz_ray_solver(initial_y - axis.y()); // y(t) = y_0
            if (t_ < 0 || t_ > 1)
                return {utils::Vector3(), INF, utils::Point2D(0, 0)};
            auto hit = bezier.getPoint(t_);
            double err = std::abs(initial_y - hit.y - axis.y());
            if (err > EPSILON_2) {
                printf("Dropping result because err too large: %lf\n", err);
                return {utils::Vector3(), INF, utils::Point2D(0, 0)};
            }
            double t = intersectSphere(ray, utils::Vector3(axis.x(), axis.y() + hit.y, axis.z()), hit.x);
            //double t = horiz_intersect(ray, t_);
            if (t < 0 || t >= INF)
                return {utils::Vector3(), INF, utils::Point2D(0, 0)};
            if (err <= EPSILON) // already accurate enough
            {
                auto pnt = ray.getVector(t);
                //printf("(%lf, %lf, %lf)\n", pnt.x(), pnt.y(), pnt.z());
                double phi = std::atan2(pnt.z() - axis.z(), pnt.x() - axis.x());
                return {pnt, t, utils::Point2D(t_, phi < 0 ? phi + PI_DOUBLED : phi)};
            } else {
                // second iteration
                t_ = horiz_ray_solver(ray.getVector(t).y() - axis.y());
                hit = bezier.getPoint(t_);
                t = intersectSphere(ray, utils::Vector3(axis.x(), axis.y() + hit.y, axis.z()), hit.x);
                //t = horiz_intersect(ray, t_);
                err = std::abs(ray.origin.y() - hit.y - axis.y());
                if (err > EPSILON_2) {
                    printf("Dropping result because err too large: %lf\n", err);
                    return {utils::Vector3(), INF, utils::Point2D(0, 0)};
                } else {
                    auto pnt = ray.getVector(t);
                    //printf("(%lf, %lf, %lf)\n", pnt.x(), pnt.y(), pnt.z());
                    double phi = std::atan2(pnt.z() - axis.z(), pnt.x() - axis.x());
                    return {pnt, t, utils::Point2D(t_, phi < 0 ? phi + PI_DOUBLED : phi)};
                }
            }
        } else // not parallel to x-z plane
        {
            auto aabb = boundingBox();
            double initial = intersectAABB(ray, aabb.first, aabb.second), initial2 = .5;
            if(initial >= INF)
                return {utils::Vector3(), INF, utils::Point2D()};
            double final_t = INF;
            double final_t_ = INF;
            auto update_final_t = [this, &final_t, &final_t_](const Ray &ray, double t_) {
                if (t_ < 0 || t_ > 1)
                    return;
                auto hit = this->bezier.getPoint(t_);
                double t = (hit.y + this->axis.y() - ray.origin.y()) / ray.direction.y();
                auto ray_hit = ray.getVector(t);
                double err = std::abs(
                        utils::Vector3(ray_hit.x() - this->axis.x(), 0, ray_hit.z() - this->axis.z()).len() - hit.x);
                if (t < final_t)
                    final_t = t, final_t_ = t_;
                if (err > EPSILON_2)
                    printf("Warning, error a bit large, %lf, current ray: o(%lf,%lf,%lf), d(%lf,%lf,%lf), hit: (%lf, %lf, %lf)\n",
                           err,
                           ray.origin.x(), ray.origin.y(), ray.origin.z(),
                           ray.direction.x(), ray.direction.y(), ray.direction.z(),
                           ray_hit.x(), ray_hit.y(), ray_hit.z());
            };

            bool traversal_order =
                    (ray.direction.y() < 0) ^bezier.sliceOrder(); // true if ray casted in the direction of increasing t
            bool flag = true;
            if (traversal_order) {
                for (size_t i = 0; i < bezier.sliceSize(); ++i) {
                    const auto &slice = bezier.getSliceParam(i);
                    double temp = (slice.first.y - (ray.origin.y() - axis.y())) / ray.direction.y();
                    auto p = ray.getVector(temp);
                    double dis = utils::Vector3(p.x() - axis.x(), 0, p.z() - axis.z()).len();
                    if (i == 0)
                        flag = dis > slice.first.x; // dis > r
                    else if (flag ^ (dis > slice.first.x)) {
                        initial = (slice.second + 2 * bezier.getSliceParam(i - 1).second) / 3;
                        initial2 = (2 * slice.second + bezier.getSliceParam(i - 1).second) / 3;
                        break;
                    }
                }
            } else {
                for (int i = bezier.sliceSize() - 1; i + 1; --i) {
                    const auto &slice = bezier.getSliceParam(i);
                    double temp = (slice.first.y - (ray.origin.y() - axis.y())) / ray.direction.y();
                    auto p = ray.getVector(temp);
                    double dis = utils::Vector3(p.x() - axis.x(), 0, p.z() - axis.z()).len();
                    if (i == bezier.sliceSize() - 1)
                        flag = dis > slice.first.x; // dis > r
                    else if (flag ^ (dis > slice.first.x)) {
                        initial = (slice.second + 2 * bezier.getSliceParam(i + 1).second) / 3;
                        initial2 = (slice.second * 2 + bezier.getSliceParam(i + 1).second) / 3;
                        break;
                    }
                }
            }

            // g(t) = [Ay(t) + B]^2 - D * x^2(t) + C
            double A = ray.direction.x() * ray.direction.x() + ray.direction.z() * ray.direction.z();
            double x0 = ray.origin.x() - axis.x(), z0 = ray.origin.z() - axis.z(), y0 = ray.origin.y() - axis.y();
            double B = (x0 * ray.direction.x() + z0 * ray.direction.z()) * ray.direction.y() - A * y0;
            double d1_sqr = ray.direction.y() * ray.direction.y();
            double C =
                    (x0 * ray.direction.z() - z0 * ray.direction.x()) *
                    (x0 * ray.direction.z() - z0 * ray.direction.x()) * d1_sqr;
            double D = d1_sqr * A;

            double t_ = normal_ray_solver(A, B, C, D, initial);
            double t2_ = normal_ray_solver(A, B, C, D, initial2);
            update_final_t(ray, t_);
            update_final_t(ray, t2_);

            if (final_t >= INF || final_t < 0)
                return {utils::Vector3(), INF, utils::Point2D(0, 0)};

            auto hit = ray.getVector(final_t);

            double phi = std::atan2(hit.z() - axis.z(), hit.x() - axis.x());
            return {hit, final_t, utils::Point2D(final_t_, phi < 0 ? phi + PI_DOUBLED : phi)};
        }
    }

    virtual std::pair<utils::Vector3, utils::Vector3> boundingBox() const override {
        return {utils::Vector3(-bezier.xMax() + axis.x() - 0.5, bezier.yMin() + axis.y() - 0.5,
                               -bezier.xMax() + axis.z() - 0.5),
                utils::Vector3(bezier.xMax() + axis.x() + 0.5, bezier.yMax() + axis.y() + 0.5,
                               bezier.xMax() + axis.z() + 0.5)};
    }

    virtual utils::Vector3 norm(const utils::Vector3 &vec, const utils::Point2D &surface_coord) const override {
        auto dd = bezier.getDerivative(surface_coord.x);
        auto tangent = utils::Vector3();
        if (std::abs(dd.y / dd.x) > 1e8)
            tangent = utils::Vector3(0, dd.y / dd.x > 0 ? 1 : -1, 0);
        else
            tangent = utils::Vector3(std::cos(surface_coord.y), dd.y / dd.x, std::sin(surface_coord.y));
        return utils::Vector3(-std::sin(surface_coord.y), 0, std::cos(surface_coord.y)).cross(
                tangent).normalize();
    }
};
