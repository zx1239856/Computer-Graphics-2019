/*
 * Created by zx on 19-4-1.
 */

#ifndef HW2_COMMON_H
#define HW2_COMMON_H
#ifndef __NVCC__

#include <cmath>
#include <limits>
#include <stdint.h>

#else
#define M_PI 3.14159265358979323846
#endif

inline double clampVal(double r, double lo = 0, double hi = 1) {
    return r < lo ? lo : r > hi ? hi : r;
}

constexpr double EPSILON = 1e-6;
constexpr double EPSILON_2 = 5e-5;
constexpr double EPSILON_1 = 1e-10;
constexpr double EPSILON_3 = 3e-3;
constexpr double EPSILON_KD = 1e-5;
constexpr double PI_DOUBLED = M_PI + M_PI;
constexpr double INF = 1e20;
constexpr int RAY_TRACING_MAX_DEPTH = 10;
constexpr int PATH_TRACING_MAX_DEPTH = 5;
constexpr double SPPM_ALPHA = 0.7;
constexpr uint32_t NULL_NODE = std::numeric_limits<uint32_t>::max();

enum SplitAxis {
    X_AXIS = 0, Y_AXIS, Z_AXIS
};

enum Refl_t {
    DIFF, SPEC, REFR
}; // material type, DIFFuse, SPECular, REFRactive

enum BRDFType{
    DIFFUSE = 0, MIRROR, GLASS, LIGHT, CERAMIC, FLOOR, WALL, METAL
};

struct BRDF {
    double specular, diffuse, refraction;
    double rho_d, rho_s, phong_s;
    double re_idx;
};

const BRDF BRDFs[] = {
        {0, 1, 0, 1, 0, 0, 0}, // DIFFUSE
        {1, 0, 0, 0, 0, 0, 0}, // MIRROR
        {0, 0, 1, 0, 0, 0, 1.65}, // GLASS
        {0, 1, 0, 1, 0, 0, 0}, // LIGHT
        {0.1, 0.9, 0, 1, 0, 0, 0}, // CERAMIC
        {0, 1, 0, 0.8, 0.2, 10, 0}, // FLOOR
        {0, 1, 0, 1, 0, 0, 0}, // WALL
        {0, 1, 0, 0, 1, 20, 0} // METAL
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
