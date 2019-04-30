#pragma once

#include <vector>
#include <cmath>
#include "../common/common.h"
#include "../common/geometry.hpp"

#ifdef __NVCC__
#include "../gpu/cuda_helpers.h"
#endif

namespace utils {

    namespace impl {
        constexpr size_t samples = 10000;

        // common impls
        template<typename arrType>
        __device__ __host__ inline Point2D __bezier_getPoint(const arrType &arr, size_t size, double t) {
            double v = 1, x = 0, y = 0;
            for (size_t i = 0; i < size; ++i) {
                x += (v * arr[i].x), y += (v * arr[i].y);
                v *= t;
            }
            return {x, y};
        }

        template<typename arrType>
        __device__ __host__ inline Point2D __bezier_getDerivative(const arrType &arr, size_t size, double t) {
            double v = 1, x = 0, y = 0;
            for (size_t i = 1; i < size; ++i) {
                x += (v * i * arr[i].x), y += (v * i * arr[i].y);
                v *= t;
            }
            return {x, y};
        }

        template<typename arrType>
        __device__ __host__ inline Point2D __bezier_getDerivative2(const arrType &arr, size_t size, double t) {
            double v = 1, x = 0, y = 0;
            for (size_t i = 2; i < size; ++i) {
                x += (v * i * (i - 1) * arr[i].x), y += (v * i * (i - 1) * arr[i].y);
                v *= t;
            }
            return {x, y};
        }
    }

#ifdef __NVCC__
    struct Bezier2D_GPU {
    public:
        KernelArray<Point2D> _ctrl_pnts;
        KernelArray<Point2D> _coeff;
        KernelArray<Point2D> _slices;
        KernelArray<double> _slices_param;
        double xmin = INF, xmax = -INF, ymin = INF, ymax = -INF;

        inline __device__ Point2D getPoint(double t) const {
           return impl::__bezier_getPoint(_coeff._array, _coeff._size, t);
        }

        inline __device__ Point2D getDerivative(double t) const {
            return impl::__bezier_getDerivative(_coeff._array, _coeff._size, t);
        }

        __device__ Point2D getDerivative2(double t) const {
            return impl::__bezier_getDerivative2(_coeff._array, _coeff._size, t);
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
#endif

    class Bezier2D {
        std::vector<Point2D> _ctrl_pnts;
        std::vector<Point2D> _coeff;
        std::vector<Point2D> _slices;
        std::vector<double> _slices_param;
        double xmin = INF, xmax = -INF, ymin = INF, ymax = -INF;
#ifdef __NVCC__
        std::vector<Bezier2D_GPU> _bezier_gpu;
#endif

        void aabbHelper() {
            size_t block = impl::samples / _ctrl_pnts.size() / 3;
            for (size_t t = 0; t < impl::samples; ++t) {
                Point2D p = getPoint(static_cast<double>(t) / impl::samples);
                if (p.x > xmax)xmax = p.x;
                if (p.x < xmin)xmin = p.x;
                if (p.y > ymax)ymax = p.y;
                if (p.y < ymin)ymin = p.y;
                if (!(t % block))
                    _slices.emplace_back(p), _slices_param.emplace_back(static_cast<double>(t) / impl::samples);
            }
            double half_block = block * .5 / impl::samples;
            if (std::abs(1 - (*_slices_param.rbegin())) >= half_block)
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

        ~Bezier2D() {
#ifdef __NVCC__
            for(auto &cu : _bezier_gpu)
            {
                releaseKernelArr(cu._ctrl_pnts);
                releaseKernelArr(cu._coeff);
                releaseKernelArr(cu._slices);
                releaseKernelArr(cu._slices_param);
            }
#endif
        }

        Point2D getPoint(double t) const {
            return impl::__bezier_getPoint(_coeff.data(), _coeff.size(), t);
        }

        Point2D getDerivative(double t) const {
            return impl::__bezier_getDerivative(_coeff.data(), _coeff.size(), t);
        }

        Point2D getDerivative2(double t) const {
            return impl::__bezier_getDerivative2(_coeff.data(), _coeff.size(), t);
        }


        inline double xMin() const {
            return xmin;
        }

        inline double xMax() const {
            return xmax;
        }

        inline double yMin() const {
            return ymin;
        }

        inline double yMax() const {
            return ymax;
        }

        inline size_t sliceSize() const {
            return _slices.size();
        }

        // return true if y in increasing order
        inline bool sliceOrder() const {
            return (*_slices.rbegin()).y > (*_slices.begin()).y;
        }

        inline std::pair<Point2D, double> getSliceParam(size_t idx) const {
            return {_slices[idx], _slices_param[idx]};
        }

        inline std::vector<Point2D> getAllCoeffs() const {
            return _coeff;
        }

        inline std::vector<Point2D> getAllSlices() const {
            return _slices;
        }

        inline std::vector<double> getAllSlicesParam() const {
            return _slices_param;
        }

#ifdef __NVCC__
        inline Bezier2D_GPU toGPU() {
            // copy CPU bezier to GPU
            Bezier2D_GPU bezier;
            bezier._ctrl_pnts = makeKernelArr(_ctrl_pnts), bezier._coeff = makeKernelArr(_coeff),
            bezier._slices = makeKernelArr(_slices), bezier._slices_param = makeKernelArr(_slices_param);
            bezier.xmax = xmax, bezier.xmin = xmin, bezier.ymax = ymax, bezier.ymin = ymin;
            _bezier_gpu.emplace_back(bezier);
            return bezier;
        }
#endif
    };
}