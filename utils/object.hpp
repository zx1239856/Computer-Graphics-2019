#pragma once

#include <vector>
#include <functional>
#include <cassert>
#include "common.h"
#include "vector3.hpp"
#include "ray.hpp"
#include "bezier.hpp"

class Texture {
public:
    utils::Vector3 color, emission;
    Refl_t refl;
    double brdf;

    Texture(const utils::Vector3 &_color, const utils::Vector3 &_emission, Refl_t _refl, double _brdf) :
            color(_color), emission(_emission), refl(_refl), brdf(_brdf) {}
};

class BasicObject {
    Texture texture;
public:
    BasicObject(const utils::Vector3 &color, const utils::Vector3 &emission, Refl_t refl, double brdf) :
            texture(color, emission, refl, brdf) {}

    BasicObject(const Texture &t) : texture(t) {}

    virtual std::pair<utils::Vector3, utils::Vector3> boundingBox() const = 0;

    virtual std::pair<utils::Vector3, double> intersect(const Ray &ray) const = 0;

    virtual utils::Vector3 norm(const utils::Vector3 &vec) const = 0;

    Texture getTexture() const { return texture; }
};

class Sphere : public BasicObject {
    utils::Vector3 origin;
    double radius;
public:
    Sphere(const utils::Vector3 &o, double r, const Texture &t) : BasicObject(t), origin(o), radius(r) {}

    Sphere(const utils::Vector3 &o, double r, const utils::Vector3 &_color, const utils::Vector3 &_emission,
           Refl_t _refl, double _brdf) :
            BasicObject(_color, _emission, _refl, _brdf), origin(o), radius(r) {}

    virtual std::pair<utils::Vector3, double> intersect(const Ray &ray) const override {
        utils::Vector3 op = origin - ray.origin;
        double b = op.dot(ray.direction);
        double d = b * b - op.len2() + radius * radius;
        if (d < 0)
            return {utils::Vector3(), INF};
        else {
            d = std::sqrt(d);
            double t = (b - d > epsilon) ? b - d : ((b + d) > epsilon ? b + d : -1);
            if (t < 0)
                return {utils::Vector3(), INF};
            return {ray.getVector(t), t};
        }
    }

    virtual std::pair<utils::Vector3, utils::Vector3> boundingBox() const override {
        return {origin - radius, origin + radius};
    }

    virtual utils::Vector3 norm(const utils::Vector3 &vec) const override {
        assert(std::abs((vec - origin).len() - radius) < epsilon);
        return (vec - origin).normalize();
    }
};

class Plane : public BasicObject {
    utils::Vector3 n;
    double d;
public:
    Plane(const utils::Vector3 &norm, double dis, const Texture &t) : BasicObject(t), n(norm.normalize()), d(dis) {}

    Plane(const utils::Vector3 &norm, double dis, const utils::Vector3 &color, const utils::Vector3 &emission,
          Refl_t refl, double brdf) :
            BasicObject(color, emission, refl, brdf), n(norm.normalize()), d(dis) {}

    virtual std::pair<utils::Vector3, double> intersect(const Ray &ray) const override {
        double prod = ray.direction.dot(n);
        double t = (d - n.dot(ray.origin)) / prod;
        if (t < epsilon)
            return {utils::Vector3(), INF};
        return {ray.getVector(t), t};
    }

    virtual std::pair<utils::Vector3, utils::Vector3> boundingBox() const override {
        return {n, n}; // the bounding box of a plane is ill-defined
    }

    virtual utils::Vector3 norm(const utils::Vector3 &vec = utils::Vector3()) const override {
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
         Refl_t refl, double brdf) :
            BasicObject(color, emission, refl, brdf), p0(min(_p0, _p1)), p1(max(_p0, _p1)) {}

    virtual std::pair<utils::Vector3, double> intersect(const Ray &ray) const override {
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
            return {ray.getVector(t_min), t_min};
        else
            return {utils::Vector3(), t_min};
    }

    virtual std::pair<utils::Vector3, utils::Vector3> boundingBox() const override {
        return {p0, p1};
    }

    virtual utils::Vector3 norm(const utils::Vector3 &vec) const override {
        if (abs(vec.x() - p0.x()) < epsilon || abs(vec.x() - p1.x()) < epsilon)
            return utils::Vector3(abs(vec.x() - p0.x()) < epsilon ? -1 : 1, 0, 0);
        if (abs(vec.y() - p0.y()) < epsilon || abs(vec.y() - p1.y()) < epsilon)
            return utils::Vector3(0, abs(vec.y() - p0.y()) < epsilon ? -1 : 1, 0);
        if (abs(vec.z() - p0.z()) < epsilon || abs(vec.z() - p1.z()) < epsilon)
            return utils::Vector3(0, 0, abs(vec.z() - p0.z()) < epsilon ? -1 : 1);
    }
};

class RotaryBezier : public BasicObject {
    utils::Vector3 axis;
    utils::Bezier2D bezier;
    // rotate bezier2d curve along y-axis (in the direction of axis vector)
public:
    RotaryBezier(const utils::Vector3 &_axis, const utils::Bezier2D &_bezier, const Texture &t) :
            BasicObject(t), axis(_axis), bezier(_bezier) {}

    RotaryBezier(const utils::Vector3 &_axis, const utils::Bezier2D &_bezier, const utils::Vector3 &color,
                 const utils::Vector3 &emission,
                 Refl_t refl, double brdf) :
            BasicObject(color, emission, refl, brdf), axis(_axis), bezier(_bezier) {}

    virtual std::pair<utils::Vector3, double> intersect(const Ray &ray) const override {

    }

    virtual std::pair<utils::Vector3, utils::Vector3> boundingBox() const override {

    }

    virtual utils::Vector3 norm(const utils::Vector3 &vec) const override {

    }
};