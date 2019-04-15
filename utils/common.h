/*
 * Created by zx on 19-4-1.
 */

#ifndef HW2_COMMON_H
#define HW2_COMMON_H
#include <cmath>

constexpr double EPSILON = 1e-6;
constexpr double EPSILON_2 = 5e-5;
constexpr double EPSILON_1 = 1e-10;
constexpr double EPSILON_3 = 5e-3;
constexpr double PI_DOUBLED = M_PI + M_PI;
constexpr double INF = 1e20;
enum Refl_t { DIFF, SPEC, REFR }; // material type, DIFFuse, SPECular, REFRactive

struct Point2D{
    double x;
    double y;
    Point2D(double x_ = 0, double y_ = 0):x(x_), y(y_){}
};

#endif //HW2_COMMON_H
