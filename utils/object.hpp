#pragma once

#include <vector>
#include <functional>
#include <cassert>
#include <cmath>
#include "bezier.hpp"
#include "../common/simple_intersect_algs.hpp"

/*
 * Common intersections(sphere, axis-aligned cubes)
 */

struct Texture {
    utils::Vector3 color, emission;
    double specular, diffuse, refraction;
    double rho_d, rho_s, phong_s;
    double re_idx;
    std::vector<std::vector<utils::Vector3>> mapped_image;
    utils::Transform2D mapped_transform;

    Texture(const utils::Vector3 &_color, const utils::Vector3 &_emission, double _specular, double _diffuse,
            double _refraction, double _rho_d, double _rho_s, double _phong_s, double _re_idx) :
            color(_color), emission(_emission), specular(_specular), diffuse(_diffuse), refraction(_refraction),
            rho_d(_rho_d), rho_s(_rho_s), phong_s(_phong_s), re_idx(_re_idx) {
        // normalization
        double s = specular + diffuse + refraction;
        specular /= s, diffuse /= s, refraction /= s;
        diffuse += specular;
        refraction += diffuse;
        s = rho_s + rho_d;
        rho_s /= s, rho_d /= s;
        rho_d += rho_s;
    }

    Texture(const utils::Vector3 &_color, const utils::Vector3 &_emission, const BRDF &brdf) :
            Texture(_color, _emission, brdf.specular, brdf.diffuse, brdf.refraction, brdf.rho_d, brdf.rho_s,
                    brdf.phong_s, brdf.re_idx) {}

    std::pair<utils::Vector3, Refl_t> getColor(const utils::Point2D &surface_coord, unsigned short X[3]) const {
        double prob = erand48(X);
        if (!mapped_image.size()) // no texture mapping. use default color
        {
            if (prob < specular)
                return {color, SPEC};
            else if (prob < diffuse)
                return {color, DIFF};
            else
                return {color, REFR};
        } else {
            const int &w = mapped_image[0].size(), &h = mapped_image.size();
            utils::Point2D p = mapped_transform.transform(surface_coord);
            int u = (lround(w * p.y + .5) % w + w) % w, v =
                    (lround(h * p.x + .5) % h + h) % h;
            auto color = mapped_image[v][u]; // 8 bit per channel, so color is in [0, 255]
            if (prob < specular)
                return {color / 255. * 0.999, SPEC};
            else if (prob < diffuse) {
                if ((color.x() >= 235 || color.y() >= 235 || color.z() >= 235) && erand48(X) < 0.13)
                    return {color / 255. * 0.999, SPEC};
                else return {color / 255. * 0.999, DIFF};
            } else
                return {color / 255. * 0.999, REFR};
        }
    }
};

/*
 * End texture section
 */

class BasicObject {
public:
    Texture texture;

    BasicObject(const utils::Vector3 &color, const utils::Vector3 &emission, const BRDF &brdf) :
            texture(color, emission, brdf) {}

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

    Sphere(const utils::Vector3 &o, double r, const utils::Vector3 &color, const utils::Vector3 &emission,
           const BRDF &brdf) :
            BasicObject(color, emission, brdf), origin(o),
            radius(r) {}

    virtual std::tuple<utils::Vector3, double, utils::Point2D> intersect(const Ray &ray) const override {
        auto &&val = intersectSphereObject(ray, origin, radius);
        return {val.first, val.second, val.third};
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

public:
    Plane(const utils::Vector3 &norm, double dis, const Texture &t) : BasicObject(t), n(norm.normalize()), d(dis) {
        preparePlaneObject(n, d, xp, yp, origin);
    }

    Plane(const utils::Vector3 &norm, double dis, const utils::Vector3 &color, const utils::Vector3 &emission,
          const BRDF &brdf) :
            BasicObject(color, emission, brdf),
            n(norm.normalize()), d(dis) {
        preparePlaneObject(n, d, xp, yp, origin);
    }

    std::tuple<utils::Vector3, double, utils::Point2D> intersect(const Ray &ray) const override {
        auto &&x = intersectPlaneObject(ray, n, d, origin, xp, yp);
        return {x.first, x.second, x.third};
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
         const utils::Vector3 &emission, const BRDF &brdf) :
            BasicObject(color, emission, brdf),
            p0(min(_p0, _p1)), p1(max(_p0, _p1)) {}

    virtual std::tuple<utils::Vector3, double, utils::Point2D> intersect(const Ray &ray) const override {
        double t = intersectAABB(ray, p0, p1);
        return {ray.getVector(t), t, utils::Point2D(0, 0)}; // surface coordinate not available for cube
    }

    virtual std::pair<utils::Vector3, utils::Vector3> boundingBox() const override {
        return {p0, p1};
    }

    virtual utils::Vector3
    norm(const utils::Vector3 &vec, const utils::Point2D &unused = utils::Point2D(0, 0)) const override {
        return normOfAABB(vec, p0, p1);
    }
};

class RotaryBezier : public BasicObject {
    utils::Vector3 axis;
    utils::Bezier2D bezier;

    // rotate bezier2d curve along y-axis
public:
    RotaryBezier(const utils::Vector3 &_axis, const utils::Bezier2D &_bezier, const Texture &t) :
            BasicObject(t), axis(_axis), bezier(_bezier) {}

    RotaryBezier(const utils::Vector3 &_axis, const utils::Bezier2D &_bezier, const utils::Vector3 &color,
                 const utils::Vector3 &emission, const BRDF &brdf) :
            BasicObject(color, emission, brdf), axis(_axis),
            bezier(_bezier) {}

    virtual std::tuple<utils::Vector3, double, utils::Point2D> intersect(const Ray &ray) const override {
        auto &&bb = boundingBox();
        auto &&x = intersectBezierObject(ray, axis, bezier, bb.first, bb.second);
        return {x.first, x.second, x.third};
    }

    virtual std::pair<utils::Vector3, utils::Vector3> boundingBox() const override {
        return {utils::Vector3(-bezier.xMax() + axis.x() - 0.5, bezier.yMin() + axis.y() - 0.5,
                               -bezier.xMax() + axis.z() - 0.5),
                utils::Vector3(bezier.xMax() + axis.x() + 0.5, bezier.yMax() + axis.y() + 0.5,
                               bezier.xMax() + axis.z() + 0.5)};
    }

    virtual utils::Vector3 norm(const utils::Vector3 &vec, const utils::Point2D &surface_coord) const override {
        return normOfBezier(vec, surface_coord, bezier);
    }
};
