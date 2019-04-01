#pragma once

#include <vector>

namespace utils {

    using Point2D = std::pair<double, double>;

    class Bezier2D {
        std::vector<Point2D> _ctrl_pnts;
        std::vector<Point2D> _coeff;

        Point2D polynomialHelper(double t, size_t st) {
            double v = 1, x = 0, y = 0;
            for (size_t i = st; i < _coeff.size(); ++i)
                x += (v * _coeff[i].first), y += (v * _coeff[i].second);
            return {x, y};
        }

    public:
        Bezier2D(const std::vector<std::pair<double, double>> &ctrl_pnts) : _ctrl_pnts(ctrl_pnts) {
            _coeff.reserve(ctrl_pnts.size());
            auto diff = ctrl_pnts;
            double nn = ctrl_pnts.size() - 1, vv = 1, v = 1;
            // v is the factorial term
            for (size_t i = 0; i < ctrl_pnts.size(); ++i) {
                _coeff[i].first = diff[0].first * v;
                _coeff[i].second = diff[0].second * v;
                for (size_t j = 0; j < ctrl_pnts.size() - i - 1; ++j)
                    _coeff[i] = {_coeff[i + 1].first - _coeff[i].first, _coeff[i + 1].second - _coeff[i].second};
                v = v * nn / vv;
                nn -= 1, vv += 1;
            }
        }

        Point2D getPoint(double t) { return polynomialHelper(t, 0); }

        Point2D getDerivative(double t) { return polynomialHelper(t, 1); }

        Point2D getDerivative2(double t) { return polynomialHelper(t, 2); }
    };
}