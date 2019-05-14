#ifndef __HW2__COMMON_SIMPLE_INTERSECT_ALGS_HPP
#define __HW2__COMMON_SIMPLE_INTERSECT_ALGS_HPP

#include "common.h"
#include "geometry.hpp"
#include "ray.hpp"

__host__ __device__ inline pair<double, utils::Vector3>
intersectTrianglularFace(const Ray &r, const utils::Vector3 &v1, const utils::Vector3 &v2, const utils::Vector3 &v3,
                         const utils::Vector3 &n1, const utils::Vector3 &n2, const utils::Vector3 &n3) {
    auto e1 = v2 - v1, e2 = v3 - v1;
    auto h = r.direction.cross(e2), s = r.origin - v1;
    double a = e1.dot(h);
    if (fabs(a) < EPSILON_2)
        return {INF, utils::Vector3()};
    double f = 1. / a;
    double u = f * h.dot(s);
    if (u < 0. || u > 1.)
        return {INF, utils::Vector3()};
    auto q = s.cross(e1);
    auto v = f * r.direction.dot(q);
    if (v < 0 || u + v > 1.)
        return {INF, utils::Vector3()};
    double t = f * e2.dot(q);
    if (t > EPSILON_2) {
        auto x = r.getVector(t);
        double s1 = (x - v2).cross(x - v3).len(), s2 = (x - v1).cross(x - v3).len(), s3 = (x - v1).cross(x - v2).len();
        auto norm = (n1 * s1 + n2 * s2 + n3 * s3).normalize();
        return {t, norm};
    } else
        return {INF, utils::Vector3()};
}

__host__ __device__ inline double intersectSphere(const Ray &ray, const utils::Vector3 &o, double r) {
    utils::Vector3 op = o - ray.origin;
    double b = op.dot(ray.direction);
    double d = b * b - op.len2() + r * r;
    if (d < 0)
        return INF;
    else {
        d = sqrt(d);
        double t = (b - d > EPSILON) ? b - d : ((b + d) > EPSILON ? b + d : -1);
        if (t < 0)
            return INF;
        return t;
    }
}

__host__ __device__ inline triplet<utils::Vector3, double, utils::Point2D>
intersectSphereObject(const Ray &ray, const utils::Vector3 &o, double r) {
    double t = intersectSphere(ray, o, r);
    if (t > 0 && t < INF) {
        auto hit = ray.getVector(t);
        double phi = std::atan2(hit.z() - o.z(), hit.x() - o.x());
        return {hit, t, utils::Point2D(acos((hit.y() - o.y()) / r),
                                       phi < 0 ? phi + PI_DOUBLED
                                               : phi)}; // spherical coordinate (theta, phi), note y is the rotation axis here
    } else
        return {utils::Vector3(), INF, utils::Point2D(0, 0)};
}

__host__ __device__ inline triplet<bool, double, double>
intersectAABBInOut(const Ray &r, const utils::Vector3 &p0, const utils::Vector3 &p1) {
    double t1 = (p0.x() - r.origin.x()) / r.direction.x();
    double t2 = (p1.x() - r.origin.x()) / r.direction.x();
    double t3 = (p0.y() - r.origin.y()) / r.direction.y();
    double t4 = (p1.y() - r.origin.y()) / r.direction.y();
    double t5 = (p0.z() - r.origin.z()) / r.direction.z();
    double t6 = (p1.z() - r.origin.z()) / r.direction.z();

    double tmin = fmax(fmax(fmin(t1, t2), fmin(t3, t4)), fmin(t5, t6));
    double tmax = fmin(fmin(fmax(t1, t2), fmax(t3, t4)), fmax(t5, t6));

    return {tmax > 0 && tmin <= tmax, tmin, tmax};
}

__host__ __device__ inline double intersectAABB(const Ray &ray, const utils::Vector3 &p0, const utils::Vector3 &p1) {
    double t, t_min = INF;

    if (ray.direction.x() > 0) // p0 is visible
        t = (p0.x() - ray.origin.x()) / ray.direction.x();
    else
        t = (p1.x() - ray.origin.x()) / ray.direction.x();

    auto v = ray.getVector(t);
    if (t > 0 && t < t_min && p0.y() <= v.y() && v.y() <= p1.y() && p0.z() <= v.z() && v.z() <= p1.z())
        t_min = t;

    if (ray.direction.y() > 0)
        t = (p0.y() - ray.origin.y()) / ray.direction.y();
    else
        t = (p1.y() - ray.origin.y()) / ray.direction.y();

    v = ray.getVector(t);
    if (t > 0 && t < t_min && p0.x() <= v.x() && v.x() <= p1.x() && p0.z() <= v.z() && v.z() <= p1.z())
        t_min = t;

    if (ray.direction.z() > 0)
        t = (p0.z() - ray.origin.z()) / ray.direction.z();
    else
        t = (p1.z() - ray.origin.z()) / ray.direction.z();

    v = ray.getVector(t);
    if (t > 0 && t < t_min && p0.y() <= v.y() && v.y() <= p1.y() && p0.x() <= v.x() && v.x() <= p1.x())
        t_min = t;

    return t_min;
}

__host__ __device__ inline triplet<utils::Vector3, double, utils::Point2D>
intersectPlaneObject(const Ray &ray, const utils::Vector3 &n, const double &d, const utils::Vector3 &origin,
                     const utils::Vector3 &xp, const utils::Vector3 &yp) {
    double prod = ray.direction.dot(n);
    double t = (d - n.dot(ray.origin)) / prod;
    if (t < EPSILON)
        return {utils::Vector3(), INF, utils::Point2D(0, 0)};
    auto &&hit = ray.getVector(t);
    auto vec = hit - origin;
    return {hit, t, utils::Point2D(vec.dot(xp), vec.dot(yp))};  // project the vector to plane
}

__host__ __device__ inline void
preparePlaneObject(const utils::Vector3 &n, const double &d, utils::Vector3 &xp, utils::Vector3 &yp,
                   utils::Vector3 &origin) {
    if (abs(abs(n.y()) - 1) < EPSILON_2)
        xp = utils::Vector3(1, 0, 0), yp = utils::Vector3(0, 0, 1);
    else
        xp = n.cross(utils::Vector3(0, 1, 0)).normalize(), yp = xp.cross(n).normalize();
    origin = n * d;
}

__host__ __device__ inline utils::Vector3
normOfAABB(const utils::Vector3 &vec, const utils::Vector3 &p0, const utils::Vector3 &p1) {
    if (abs(vec.x() - p0.x()) < EPSILON || abs(vec.x() - p1.x()) < EPSILON)
        return utils::Vector3(abs(vec.x() - p0.x()) < EPSILON ? -1 : 1, 0, 0);
    if (abs(vec.y() - p0.y()) < EPSILON || abs(vec.y() - p1.y()) < EPSILON)
        return utils::Vector3(0, abs(vec.y() - p0.y()) < EPSILON ? -1 : 1, 0);
    if (abs(vec.z() - p0.z()) < EPSILON || abs(vec.z() - p1.z()) < EPSILON)
        return utils::Vector3(0, 0, abs(vec.z() - p0.z()) < EPSILON ? -1 : 1);
    return utils::Vector3();
}

// bezier utils
template<typename BezierType>
__host__ __device__ inline double bezier_horiz_ray_solver(double y, const BezierType &bezier, size_t iter = 15) {
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

template<typename BezierType>
__host__ __device__ inline double
bezier_normal_ray_solver(double A, double B, double C, double D, double initial_val, const BezierType &bezier,
                         size_t iter = 15) {
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

template<typename BezierType>
__host__ __device__ inline utils::Vector3
normOfBezier(const utils::Vector3 &vec, const utils::Point2D &surface_coord, const BezierType &bezier) {
    auto dd = bezier.getDerivative(surface_coord.x);
    auto tangent = utils::Vector3();
    if (abs(dd.y / dd.x) > 1e8)
        tangent = utils::Vector3(0, dd.y / dd.x > 0 ? 1 : -1, 0);
    else
        tangent = utils::Vector3(cos(surface_coord.y), dd.y / dd.x, sin(surface_coord.y));
    return utils::Vector3(-sin(surface_coord.y), 0, cos(surface_coord.y)).cross(
            tangent).normalize();
}

template<typename BezierType>
__host__ __device__ void
bezier_update_final_t(const Ray &ray, double t_, double &final_t, double &final_t_, const BezierType &bezier,
                      const utils::Vector3 &axis) {
    if (t_ < 0 || t_ > 1)
        return;
    auto hit = bezier.getPoint(t_);
    double t = (hit.y + axis.y() - ray.origin.y()) / ray.direction.y();
    auto ray_hit = ray.getVector(t);
    if (t < final_t)
        final_t = t, final_t_ = t_;
}

template<typename BezierType>
__host__ __device__ triplet<utils::Vector3, double, utils::Point2D>
intersectBezierObject(const Ray &ray, const utils::Vector3 &axis, const BezierType &bezier, const utils::Vector3 &p0,
                      utils::Vector3 &p1) {
    if (abs(ray.direction.y()) < 3e-3) { // light parallel to x-z plane
        double temp = utils::Vector3(axis.x() - ray.origin.x(), 0, axis.z() - ray.origin.z()).len();
        double initial_y = ray.getVector(temp).y();
        double t_ = bezier_horiz_ray_solver(initial_y - axis.y(), bezier); // y(t) = y_0
        if (t_ < 0 || t_ > 1)
            return {utils::Vector3(), INF, utils::Point2D(0, 0)};
        auto hit = bezier.getPoint(t_);
        double err = abs(initial_y - hit.y - axis.y());
        if (err > EPSILON_2) {
            // drop value because error too large
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
            double phi = atan2(pnt.z() - axis.z(), pnt.x() - axis.x());
            return {pnt, t, utils::Point2D(t_, phi < 0 ? phi + PI_DOUBLED : phi)};
        } else {
            // second iteration
            t_ = bezier_horiz_ray_solver(ray.getVector(t).y() - axis.y(), bezier);
            hit = bezier.getPoint(t_);
            t = intersectSphere(ray, utils::Vector3(axis.x(), axis.y() + hit.y, axis.z()), hit.x);
            //t = horiz_intersect(ray, t_);
            err = abs(ray.origin.y() - hit.y - axis.y());
            if (err > EPSILON_2) {
                //printf("Dropping result because err too large: %lf\n", err);
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
        double initial = intersectAABB(ray, p0, p1), initial2 = .5;
        if (initial >= INF)
            return {utils::Vector3(), INF, utils::Point2D()};
        double final_t = INF;
        double final_t_ = INF;

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

        double t_ = bezier_normal_ray_solver(A, B, C, D, initial, bezier);
        double t2_ = bezier_normal_ray_solver(A, B, C, D, initial2, bezier);
        bezier_update_final_t(ray, t_, final_t, final_t_, bezier, axis);
        bezier_update_final_t(ray, t2_, final_t, final_t_, bezier, axis);

        if (final_t >= INF || final_t < 0)
            return {utils::Vector3(), INF, utils::Point2D(0, 0)};

        auto hit = ray.getVector(final_t);

        double phi = atan2(hit.z() - axis.z(), hit.x() - axis.x());
        return {hit, final_t, utils::Point2D(final_t_, phi < 0 ? phi + PI_DOUBLED : phi)};
    }
}

#endif