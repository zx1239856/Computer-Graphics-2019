#pragma once

#include <vector>
#include <cassert>
#include "common.h"
#include "vector3.hpp"
#include "ray.hpp"

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

    Texture getTexture() const {return texture; }
};

class Sphere : public BasicObject {
    utils::Vector3 origin;
    double radius;
public:
    Sphere(const utils::Vector3 &o, double r, Texture t) : BasicObject(t), origin(o), radius(r) {}

    Sphere(const utils::Vector3 &o, double r, const utils::Vector3 &_color, const utils::Vector3 &_emission,
           Refl_t _refl, double _brdf) :
            BasicObject(_color, _emission, _refl, _brdf), origin(o), radius(r) {}
    virtual std::pair<utils::Vector3, double> intersect(const Ray &ray) const override
    {
        auto op = origin - ray.origin;
        double b = op.dot(ray.direction);
        double d = b * b - op.len2() + radius * radius;
        if(d < 0)
            return {utils::Vector3(), INF};
        else
        {
            d = std::sqrt(d);
            double t = (b - d > epsilon) ? b - d : ((b + d) > epsilon ? b + d : 0);
            return {ray.getVector(t), t};
        }
    }
    virtual std::pair<utils::Vector3, utils::Vector3> boundingBox() const override
    {
        return { origin - radius, origin + radius };
    }
    virtual utils::Vector3 norm(const utils::Vector3 &vec) const override
    {
        assert(std::abs((vec - origin).len() - radius) < epsilon);
        return (vec - origin).normalize();
    }
};