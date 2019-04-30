/*
 * Created by zx on 19-4-1.
 */

#ifndef HW2_COMMON_H
#define HW2_COMMON_H
#ifndef __NVCC__
#include <cmath>
#else
#define M_PI 3.14159265358979323846
#endif

constexpr double EPSILON = 1e-6;
constexpr double EPSILON_2 = 5e-5;
constexpr double EPSILON_1 = 1e-10;
constexpr double EPSILON_3 = 3e-3;
constexpr double PI_DOUBLED = M_PI + M_PI;
constexpr double INF = 1e20;
enum Refl_t { DIFF, SPEC, REFR }; // material type, DIFFuse, SPECular, REFRactive

template <typename T, typename K>
struct pair
{
    T first;
    K second;
};

template <typename T, typename K, typename L>
struct triplet
{
    T first;
    K second;
    L third;
};

#endif //HW2_COMMON_H
