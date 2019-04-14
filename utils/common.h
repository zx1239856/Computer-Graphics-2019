/*
 * Created by zx on 19-4-1.
 */

#ifndef HW2_COMMON_H
#define HW2_COMMON_H

constexpr double epsilon = 1e-6;
constexpr double epsilon2 = 5e-5;
constexpr double INF = 1e20;
enum Refl_t { DIFF, SPEC, REFR }; // material type, DIFFuse, SPECular, REFRactive

struct Point2D{
    double x;
    double y;
    Point2D(double x_ = 0, double y_ = 0):x(x_), y(y_){}
};

#endif //HW2_COMMON_H
