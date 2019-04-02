#ifndef _VECTOR_3_HPP
#define _VECTOR_3_HPP

#include <cmath>
#include <algorithm>

namespace utils {
    class Vector3 {
        using FloatType = double;
        FloatType _x, _y, _z;

        double clampHelper(FloatType lo, FloatType hi, FloatType v) {
            return v < lo ? lo : (v > hi ? hi : v);
        }

    public:
        explicit Vector3(FloatType x = 0, FloatType y = 0, FloatType z = 0) : _x(x), _y(y), _z(z) {}

        FloatType x() const { return _x; }

        FloatType y() const { return _y; }

        FloatType z() const { return _z; }

        Vector3 mult(const Vector3 &other) const { return Vector3(_x * other._x, _y * other._y, _z * other._z); }

        Vector3 operator+(const Vector3 &other) const { return Vector3(_x + other._x, _y + other._y, _z + other._z); }

        Vector3 operator+(FloatType v) const { return Vector3(_x + v, _y + v, _z + v); }

        Vector3 operator-(const Vector3 &other) const { return Vector3(_x - other._x, _y - other._y, _z - other._z); }

        Vector3 operator-() const { return Vector3(-_x, -_y, -_z); }

        Vector3 operator-(FloatType v) const { return Vector3(_x - v, _y - v, _z - v); }

        Vector3 operator*(FloatType v) const { return Vector3(_x * v, _y * v, _z * v); }

        Vector3 operator*(const Vector3 &other) { return Vector3(_x * other._x, _y * other._y, _z * other._z); }

        Vector3 operator/(FloatType v) const { return Vector3(_x / v, _y / v, _z / v); }

        Vector3 &operator+=(const Vector3 &other) { return *this = *this + other; }

        Vector3 &operator+=(FloatType v) { return *this = *this + v; }

        Vector3 &operator-=(const Vector3 &other) { return *this = *this - other; }

        Vector3 &operator-=(FloatType v) { return *this = *this - v; }

        Vector3 &operator*=(FloatType v) { return *this = *this * v; }

        Vector3 &operator/=(FloatType v) { return *this = *this / v; }

        FloatType dot(const Vector3 &other) const { return _x * other._x + _y * other._y + _z * other._z; }

        Vector3 cross(const Vector3 &other) const {
            return Vector3(_y * other._z - _z * other._y, _z * other._x - _x * other._z, _x * other._y - _y * other._x);
        }

        FloatType len() const { return std::sqrt(_x * _x + _y * _y + _z * _z); }

        FloatType len2() const { return _x * _x + _y * _y + _z * _z; }

        Vector3 normalize() const { return (*this) / len(); }

        double max() const { return (_x > _y && _x > _z) ? _x : _y > _z ? _y : _z; }

        double min() const { return (_x < _y && _x < _z) ? _x : _y < _z ? _y : _z; }

        Vector3 reflect(const Vector3 &n) const { return (*this) - n * n.dot(*this) * 2.; }

        Vector3 refract(const Vector3 &n, FloatType ni, FloatType nr) const {
            if (ni == nr) return *this;
            FloatType cosi = this->normalize().dot(n);
            FloatType n_ir = ni / nr;
            FloatType cosr2 = 1. - n_ir * n_ir * (1 - cosi * cosi);
            if (cosr2 <= 0) return Vector3(); // complete reflection
            FloatType cosr = (cosi > 0) ? std::sqrt(cosr2) : -std::sqrt(cosr2);
            return ((*this) * n_ir + n * (-n_ir * cosi + cosr)).normalize();
        }

        Vector3 clamp(FloatType lo, FloatType hi) {
            return Vector3(clampHelper(lo, hi, this->_x), clampHelper(lo, hi, this->_y), clampHelper(lo, hi, this->_z));
        }

        friend Vector3 min(const Vector3 &a, const Vector3 &b);

        friend Vector3 max(const Vector3 &a, const Vector3 &b);
    };

    inline Vector3 min(const Vector3 &a, const Vector3 &b) {
        return Vector3(std::min(a._x, b._x), std::min(a._y, b._y), std::min(a._z, b._z));
    }

    inline Vector3 max(const Vector3 &a, const Vector3 &b) {
        return Vector3(std::max(a._x, b._x), std::max(a._y, b._y), std::max(a._z, b._z));
    }
}

#endif