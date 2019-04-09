#pragma once

#include <vector>
#include "common.h"

namespace utils {

    using Point2D = std::pair<double, double>;

    constexpr size_t samples = 10000;

    class Bezier2D {
        std::vector<Point2D> _ctrl_pnts;
        std::vector<Point2D> _coeff;
        double xmin = INF, xmax = -INF, ymin = INF, ymax = -INF;

        void aabbHelper() {
            for (size_t t = 0; t < samples; ++t) {
                Point2D p = getPoint(static_cast<double>(t) / samples);
                if (p.first > xmax)xmax = p.first;
                if (p.first < xmin)xmin = p.first;
                if (p.second > ymax)ymax = p.second;
                if (p.second < ymin)ymin = p.second;
            }
        }

    public:
        Bezier2D(const std::vector<std::pair<double, double>> &ctrl_pnts) : _ctrl_pnts(ctrl_pnts),
                                                                            _coeff(ctrl_pnts.size()) {
            auto diff = ctrl_pnts;
            double nn = ctrl_pnts.size() - 1, vv = 1, v = 1;
            // v is the factorial term
            for (size_t i = 0; i < ctrl_pnts.size(); ++i) {
                _coeff[i].first = diff[0].first * v;
                _coeff[i].second = diff[0].second * v;
                // differential points
                for (size_t j = 0; j < ctrl_pnts.size() - i - 1; ++j)
                    diff[j] = {diff[j + 1].first - diff[j].first, diff[j + 1].second - diff[j].second};
                v = v * nn / vv;
                nn -= 1, vv += 1;
            }
            // discrete sample to calculate the aabb
            aabbHelper();
        }

        Point2D getPoint(double t) const {
            double v = 1, x = 0, y = 0;
            for (size_t i = 0; i < _coeff.size(); ++i) {
                x += (v * _coeff[i].first), y += (v * _coeff[i].second);
                v *= t;
            }
            return {x, y};
        }

        Point2D getDerivative(double t) const {
            double v = 1, x = 0, y = 0;
            for (size_t i = 1; i < _coeff.size(); ++i) {
                x += (v * i * _coeff[i].first), y += (v * i * _coeff[i].second);
                v *= t;
            }
            return {x, y};
        }

        Point2D getDerivative2(double t) const {
            double v = 1, x = 0, y = 0;
            for (size_t i = 2; i < _coeff.size(); ++i) {
                x += (v * i * (i - 1) * _coeff[i].first), y += (v * i * (i - 1) * _coeff[i].second);
                v *= t;
            }
            return {x, y};
        }


        double xMin() const {
            return xmin;
        }

        double xMax() const {
            return xmax;
        }

        double yMin() const {
            return ymin;
        }

        double yMax() const {
            return ymax;
        }
    };
}