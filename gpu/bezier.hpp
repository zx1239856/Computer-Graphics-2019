#pragma once

#include "../common/common.h"
#include "../common/geometry.hpp"
#include "cuda_helpers.h"

namespace utils {
    /*
     * Usage: Depend on CPU version, calculate all params first then copy to this class
     */

    class Bezier2D_GPU {
    public:
        KernelArray<Point2D> _ctrl_pnts;
        KernelArray<Point2D> _coeff;
        KernelArray<Point2D> _slices;
        KernelArray<double> _slices_param;
        double xmin = INF, xmax = -INF, ymin = INF, ymax = -INF;

        __device__ Point2D getPoint(double t) const {
            double v = 1, x = 0, y = 0;
            for (size_t i = 0; i < _coeff._size; ++i) {
                x += (v * _coeff._array[i].x), y += (v * _coeff._array[i].y);
                v *= t;
            }
            return {x, y};
        }

        __device__ Point2D getDerivative(double t) const {
            double v = 1, x = 0, y = 0;
            for (size_t i = 1; i < _coeff._size; ++i) {
                x += (v * i * _coeff._array[i].x), y += (v * i * _coeff._array[i].y);
                v *= t;
            }
            return {x, y};
        }

        __device__ Point2D getDerivative2(double t) const {
            double v = 1, x = 0, y = 0;
            for (size_t i = 2; i < _coeff._size; ++i) {
                x += (v * i * (i - 1) * _coeff._array[i].x), y += (v * i * (i - 1) * _coeff._array[i].y);
                v *= t;
            }
            return {x, y};
        }

        __device__ size_t sliceSize() const {
            return _slices._size;
        }

        // return true if y in increasing order
        __device__ bool sliceOrder() const {
            return _slices._array[_slices._size - 1].y > _slices._array[0].y;
        }

        __device__ pair<Point2D, double> getSliceParam(size_t idx) const {
            return {_slices._array[idx], _slices_param._array[idx]};
        }
    };
}