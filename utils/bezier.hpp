#pragma once

#include <vector>
#include <cmath>
#include "common.h"

namespace utils {

    constexpr size_t samples = 10000;

    class Bezier2D {
        std::vector<Point2D> _ctrl_pnts;
        std::vector<Point2D> _coeff;
        std::vector<Point2D> _slices;
        std::vector<double> _slices_param;
        double xmin = INF, xmax = -INF, ymin = INF, ymax = -INF;

        void aabbHelper() {
            size_t block = samples / _ctrl_pnts.size();
            for (size_t t = 0; t < samples; ++t) {
                Point2D p = getPoint(static_cast<double>(t) / samples);
                if (p.x > xmax)xmax = p.x;
                if (p.x < xmin)xmin = p.x;
                if (p.y > ymax)ymax = p.y;
                if (p.y < ymin)ymin = p.y;
                if(!(t % block))
                    _slices.emplace_back(p), _slices_param.emplace_back(static_cast<double>(t) / samples);
            }
            double half_block = block * .5 / samples;
            if(std::abs(1 - (*_slices_param.rbegin())) >= half_block)
                _slices.push_back(getPoint(1)), _slices_param.emplace_back(1);
        }

    public:
        Bezier2D(const std::vector<Point2D> &ctrl_pnts) : _ctrl_pnts(ctrl_pnts),
                                                                            _coeff(ctrl_pnts.size()) {
            auto diff = ctrl_pnts;
            double nn = ctrl_pnts.size() - 1, vv = 1, v = 1;
            // v is the factorial term
            for (size_t i = 0; i < ctrl_pnts.size(); ++i) {
                _coeff[i].x = diff[0].x * v;
                _coeff[i].y = diff[0].y * v;
                // differential points
                for (size_t j = 0; j < ctrl_pnts.size() - i - 1; ++j)
                    diff[j] = {diff[j + 1].x - diff[j].x, diff[j + 1].y - diff[j].y};
                v = v * nn / vv;
                nn -= 1, vv += 1;
            }
            // discrete sample to calculate the aabb
            aabbHelper();
        }

        Point2D getPoint(double t) const {
            double v = 1, x = 0, y = 0;
            for (size_t i = 0; i < _coeff.size(); ++i) {
                x += (v * _coeff[i].x), y += (v * _coeff[i].y);
                v *= t;
            }
            return {x, y};
        }

        Point2D getDerivative(double t) const {
            double v = 1, x = 0, y = 0;
            for (size_t i = 1; i < _coeff.size(); ++i) {
                x += (v * i * _coeff[i].x), y += (v * i * _coeff[i].y);
                v *= t;
            }
            return {x, y};
        }

        Point2D getDerivative2(double t) const {
            double v = 1, x = 0, y = 0;
            for (size_t i = 2; i < _coeff.size(); ++i) {
                x += (v * i * (i - 1) * _coeff[i].x), y += (v * i * (i - 1) * _coeff[i].y);
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

        size_t sliceSize() const {
            return _slices.size();
        }

        // return true if y in increasing order
        bool sliceOrder() const {
            return (*_slices.rbegin()).y > (*_slices.begin()).y;
        }

        std::pair<Point2D, double> getSliceParam(size_t idx) const {
            return {_slices[idx], _slices_param[idx]};
        }
    };
}