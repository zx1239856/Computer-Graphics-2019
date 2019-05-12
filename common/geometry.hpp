#ifndef _VECTOR_3_HPP
#define _VECTOR_3_HPP

#ifndef __NVCC__

#include <cmath>
#include <algorithm>

#define __device__
#define __host__
#endif

namespace utils {

    template<typename T>
    __device__ __host__ inline T min(const T &a, const T &b) {
        return a < b ? a : b;
    }

    template<typename T>
    __device__ __host__ inline T max(const T &a, const T &b) {
        return a < b ? b : a;
    }

    class Vector3 {
        using FloatType = double;
        FloatType _x, _y, _z;

        __device__ __host__ double clampHelper(FloatType lo, FloatType hi, FloatType v) {
            return v < lo ? lo : (v > hi ? hi : v);
        }

    public:
        __device__ __host__ explicit Vector3(FloatType x = 0, FloatType y = 0, FloatType z = 0) : _x(x), _y(y), _z(z) {}

        __device__ __host__ FloatType x() const { return _x; }

        __device__ __host__ FloatType y() const { return _y; }

        __device__ __host__ FloatType z() const { return _z; }

        __device__ __host__ FloatType &x() { return _x; }

        __device__ __host__ FloatType &y() { return _y; }

        __device__ __host__ FloatType &z() { return _z; }

        __device__ __host__ Vector3 mult(const Vector3 &other) const {
            return Vector3(_x * other._x, _y * other._y, _z * other._z);
        }

        __device__ __host__ Vector3 operator+(const Vector3 &other) const {
            return Vector3(_x + other._x, _y + other._y, _z + other._z);
        }

        __device__ __host__ Vector3 operator+(FloatType v) const { return Vector3(_x + v, _y + v, _z + v); }

        __device__ __host__ Vector3 operator-(const Vector3 &other) const {
            return Vector3(_x - other._x, _y - other._y, _z - other._z);
        }

        __device__ __host__ Vector3 operator-() const { return Vector3(-_x, -_y, -_z); }

        __device__ __host__ Vector3 operator-(FloatType v) const { return Vector3(_x - v, _y - v, _z - v); }

        __device__ __host__ Vector3 operator*(FloatType v) const { return Vector3(_x * v, _y * v, _z * v); }

        __device__ __host__ Vector3 operator*(const Vector3 &other) {
            return Vector3(_x * other._x, _y * other._y, _z * other._z);
        }

        __device__ __host__ Vector3 operator/(FloatType v) const { return Vector3(_x / v, _y / v, _z / v); }

        __device__ __host__ Vector3 &operator+=(const Vector3 &other) { return *this = *this + other; }

        __device__ __host__ Vector3 &operator+=(FloatType v) { return *this = *this + v; }

        __device__ __host__ Vector3 &operator-=(const Vector3 &other) { return *this = *this - other; }

        __device__ __host__ Vector3 &operator-=(FloatType v) { return *this = *this - v; }

        __device__ __host__ Vector3 &operator*=(FloatType v) { return *this = *this * v; }

        __device__ __host__ Vector3 &operator/=(FloatType v) { return *this = *this / v; }

        __device__ __host__ FloatType dot(const Vector3 &other) const {
            return _x * other._x + _y * other._y + _z * other._z;
        }

        __device__ __host__ Vector3 cross(const Vector3 &other) const {
            return Vector3(_y * other._z - _z * other._y, _z * other._x - _x * other._z, _x * other._y - _y * other._x);
        }

        __device__ __host__ FloatType len() const { return sqrt(_x * _x + _y * _y + _z * _z); }

        __device__ __host__ FloatType len2() const { return _x * _x + _y * _y + _z * _z; }

        __device__ __host__ Vector3 normalize() const { return (*this) / len(); }

        __device__ __host__ double max() const { return (_x > _y && _x > _z) ? _x : _y > _z ? _y : _z; }

        __device__ __host__ double min() const { return (_x < _y && _x < _z) ? _x : _y < _z ? _y : _z; }

        __device__ __host__ bool operator==(const Vector3 &other) const {
            return _x == other._x && _y == other._y && _z == other._z;
        }

        __device__ __host__ bool operator!=(const Vector3 &other) const {
            return _x != other._x || _y != other._y || _z != other._z;
        }

        __device__ __host__ Vector3 reflect(const Vector3 &n) const { return (*this) - n * n.dot(*this) * 2.; }

        __device__ __host__ Vector3 refract(const Vector3 &n, FloatType ni, FloatType nr) const {
            if (ni == nr) return *this;
            FloatType cosi = this->normalize().dot(n);
            FloatType n_ir = ni / nr;
            FloatType cosr2 = 1. - n_ir * n_ir * (1 - cosi * cosi);
            if (cosr2 <= 0) return Vector3(); // complete reflection
            FloatType cosr = (cosi > 0) ? sqrt(cosr2) : -sqrt(cosr2);
            return ((*this) * n_ir + n * (-n_ir * cosi + cosr)).normalize();
        }

        __device__ __host__ Vector3 clamp(FloatType lo, FloatType hi) {
            return Vector3(clampHelper(lo, hi, this->_x), clampHelper(lo, hi, this->_y), clampHelper(lo, hi, this->_z));
        }
    };

    template<>
    __device__ __host__ inline Vector3 min<Vector3>(const Vector3 &a, const Vector3 &b) {
        return Vector3(min(a.x(), b.x()), min(a.y(), b.y()), min(a.z(), b.z()));
    }

    template<>
    __device__ __host__ inline Vector3 max<Vector3>(const Vector3 &a, const Vector3 &b) {
        return Vector3(max(a.x(), b.x()), max(a.y(), b.y()), max(a.z(), b.z()));
    }

    struct Point2D {
        double x;
        double y;

        __device__ __host__ Point2D(double x_ = 0, double y_ = 0) : x(x_), y(y_) {}

        __device__ __host__ Point2D operator+(const Point2D &other) const {
            return Point2D(x + other.x, y + other.y);
        }

        __device__ __host__ Point2D operator-(const Point2D &other) const {
            return Point2D(x - other.x, y - other.y);
        }

        __device__ __host__ double distance(const Point2D &other) const {
            return sqrt((x - other.x) * (x - other.x) + (y - other.y) * (y - other.y));
        }

        __device__ __host__ double distance2(const Point2D &other) const {
            return (x - other.x) * (x - other.x) + (y - other.y) * (y - other.y);
        }
    };

    struct Transform2D {
        double x11, x12, x21, x22;
        double tx, ty;

        __device__ __host__ explicit Transform2D(double _x11 = 1, double _x12 = 0, double _x21 = 0, double _x22 = 1,
                                                 double _tx = 0, double _ty = 0)
                : x11(_x11), x12(_x12),
                  x21(_x21), x22(_x22), tx(_tx), ty(_ty) {}

        __device__ __host__ Point2D transform(const Point2D &p) const {
            return Point2D(x11 * p.x + x12 * p.y + tx, x21 * p.x + x22 * p.y + ty);
        }
    };
}

#endif