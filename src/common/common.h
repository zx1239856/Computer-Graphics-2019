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
constexpr double EPSILON_KD = 1e-5;
constexpr double PI_DOUBLED = M_PI + M_PI;
constexpr double INF = 1e20;
enum Refl_t {
    DIFF, SPEC, REFR
}; // material type, DIFFuse, SPECular, REFRactive

enum {
    DIFFUSE = 0, MIRROR, GLASS, LIGHT, CERAMIC, FLOOR, WALL, PHONG
};

struct BRDF {
    double specular, diffuse, refraction;
    double rho_d, rho_s, phong_s;
    double re_idx;
};

const BRDF BRDFs[] = {
        {0, 1, 0, 0.7, 0, 0, 0}, // DIFFUSE
        {1, 0, 0, 0, 0, 0, 0}, // MIRROR
        {0, 0, 1, 0, 0, 0, 1.65}, // GLASS
        {0, 1, 0, 0, 0, 0, 0}, // LIGHT
        {0.1, 0.9, 0, 1, 0, 30, 0}, // CERAMIC
        {0.1, 0.9, 0, 0.9, 0.1, 50, 0}, // FLOOR
        {0, 1, 0, 1, 0, 0, 0}, // WALL
        {0, 1, 0, 0.8, 0.2, 0.3, 0} // PHONG
};

template<typename T, typename K>
struct pair {
    T first;
    K second;
};

template<typename T, typename K, typename L>
struct triplet {
    T first;
    K second;
    L third;
};

struct TriangleFace {
    int v1, v2, v3, n1, n2, n3;
    int idx;
};

#endif //HW2_COMMON_H
