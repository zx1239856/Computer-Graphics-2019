#include <cuda_runtime.h>
#include "cuda_helpers.h"
#include "math_helpers.h"
#include "object.hpp"
#include "render_helpers.h"
#include "../utils/bezier.hpp"
#include "../utils/kdtree.hpp"
#include "../obj/obj_wrapper.h"
#include <cuda_profiler_api.h>

__global__ void
debug_kernel(KernelArray<Sphere_GPU> spheres, KernelArray<Cube_GPU> cubes, KernelArray<Plane_GPU> planes,
             KernelArray<RotaryBezier_GPU> beziers, KernelArray<TriangleMeshObject_GPU> meshes) {
    Ray r(utils::Vector3(0, 50, 100), utils::Vector3(1, -0.1, -1).normalize());
    printf("Total count: #sphere: %d, #cube: %d, #plane: %d, #bezier: %d\n", spheres._size, cubes._size, planes._size,
           beziers._size);
    for (auto i = 0; i < spheres._size; ++i)
        spheres._array[i].intersect(r);
    for (auto i = 0; i < cubes._size; ++i)
        cubes._array[i].intersect(r);
    for (auto i = 0; i < planes._size; ++i)
        planes._array[i].intersect(r);
    for (auto i = 0; i < spheres._size; ++i)
        beziers._array[i].intersect(r);
    for (auto i = 0; i < meshes._size; ++i)
        meshes._array[i].intersect(r);
}

__device__ static utils::Vector3
radiance(KernelArray<Sphere_GPU> &spheres, KernelArray<Cube_GPU> &cubes, KernelArray<Plane_GPU> &planes,
         KernelArray<RotaryBezier_GPU> &beziers, KernelArray<TriangleMeshObject_GPU> &meshes, const Ray &ray,
         curandState *state) {
    Ray r = ray;
    utils::Vector3 L, F(1., 1., 1.);
    size_t depth = 0;
    while (true) {
        auto &&res = findFirstIntersect(planes, cubes, spheres, beziers, meshes,
                                        r);  // triplet<utils::Vector3, double, pair<utils::Point2D, Texture_GPU*>>
        auto &&texture = res.third.second;
        if (res.second >= INF || res.second < 0)
            return L;
        auto &&prop = res.third.second->getColor(res.third.first, state); // color, refl_t
        L += F.mult(res.third.second->emission);
        F = F.mult(prop.first);
        double p = prop.first.max();

        bool into = false;
        utils::Vector3 x = r.getVector(res.second), nl =
                res.first.dot(r.direction) < 0 ? into = true, res.first : -res.first;
        Ray reflray = Ray(x, r.direction.reflect(nl));

        switch (prop.second) {
            case REFR: {
                utils::Vector3 d = r.direction.refract(res.first, into ? 1 : res.third.second->re_idx,
                                                       into ? res.third.second->re_idx : 1);
                if (d.len2() < EPSILON) // total internal reflection
                {
                    r = reflray;
                    continue;
                }
                double a = res.third.second->re_idx - 1, b = res.third.second->re_idx + 1, R0 = a * a / (b * b), c =
                        1 - (into ? -r.direction.dot(nl) : d.dot(res.first));
                double Re = R0 + (1 - R0) * c * c * c * c * c, Tr = 1 - Re, P = .25 + .5 * Re, RP = Re / P, TP =
                        Tr / (1 - P);
                if (curand_uniform_double(state) < P) {
                    F *= RP;
                    r = reflray;
                    continue;
                } else {
                    F *= TP;
                    r = Ray(x, d);
                    continue;
                }
            }
            case SPEC: {
                r = reflray;
                break;
            }
            default: {
                double a = curand_uniform_double(state);
                if (a < texture->rho_s) {
                    // phong specular
                    double phi = 2 * M_PI * curand_uniform_double(state), r2 = curand_uniform_double(state);
                    double cos_theta = pow(1 - r2, 1 / (1 + texture->phong_s));
                    double sin_theta = sqrt(1 - cos_theta * cos_theta);
                    utils::Vector3 w = reflray.direction, u = ((fabs(w.x()) > .1 ? utils::Vector3(0, 1)
                                                                                         : utils::Vector3(1)).cross(
                            w)).normalize(), v = w.cross(u).normalize();
                    utils::Vector3 d = (u * cos(phi) * sin_theta + v * sin(phi) * sin_theta +
                                        w * cos_theta).normalize();
                    r = Ray(x, d);
                } else if (a < texture->rho_d) {
                    double r1 = 2 * M_PI * curand_uniform_double(state), r2s = sqrt(curand_uniform_double(state));
                    utils::Vector3 w = nl, u = ((fabs(w.x()) > .1 ? utils::Vector3(0, 1) : utils::Vector3(1)).cross(
                            w)).normalize(), v = w.cross(u).normalize();
                    utils::Vector3 d = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2s * r2s)).normalize();
                    r = Ray(x, d);
                } else return L;
            }
                break;
        }
	if (++depth > PATH_TRACING_MAX_DEPTH) {
            if (curand_uniform_double(state) < p)
                F /= p;
            else return L;
        }
    }
}

__global__ void render_pt(KernelArray<Sphere_GPU> spheres, KernelArray<Cube_GPU> cubes, KernelArray<Plane_GPU> planes,
                          KernelArray<RotaryBezier_GPU> beziers, KernelArray<TriangleMeshObject_GPU> meshes, Camera cam,
                          int samps,
                          KernelArray<utils::Vector3> out_buffer) {
    const uint32_t u = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t v = threadIdx.y + blockIdx.y * blockDim.y;
    const uint32_t pos = u + v * blockDim.x * gridDim.x;
    if (u >= cam.w || v >= cam.h) {
        return;
    }
    curandState state;
    curand_init(pos, 0u, 0u, &state);

    utils::Vector3 cx = utils::Vector3(-cam.direction.z(), 0, cam.direction.x());
    if (cx.len2() < EPSILON) {
        cx = utils::Vector3(1, 0, 0) * cam.w / cam.h * cam.fov;  // camera pointing towards y-axis
    } else cx = cx.normalize() * cam.w / cam.h * cam.fov;
    utils::Vector3 cy = cx.cross(cam.direction).normalize() * cam.fov;

    for (int sy = 0; sy < 2; ++sy) {  // 2x2 super sampling
        for (int sx = 0; sx < 2; ++sx) {
            utils::Vector3 r = utils::Vector3();
            for (int s = 0; s < samps; ++s) {
                double r1 = 2 * curand_uniform_double(&state), dx = r1 < 1 ? sqrtf(r1) : 2 - sqrtf(2 - r1);
                double r2 = 2 * curand_uniform_double(&state), dy = r2 < 1 ? sqrtf(r2) : 2 - sqrtf(2 - r2);
                utils::Vector3 d = cx * (((sx + .5 + dx) / 2 + u) / cam.w - .5) +
                                   cy * (((sy + .5 + dy) / 2 + v) / cam.h - .5) + cam.direction;
                double cos = d.normalize().dot(cam.direction);
                utils::Vector3 hit = cam.origin + d * cam.focal_dist / cos;  // real hit point on focal plane
                double theta = curand_uniform_double(&state) * M_PI * 2;
                utils::Vector3 p_origin = cam.origin +
                                              (cx.normalize() * std::cos(theta) + cy.normalize() * std::sin(theta)) * curand_uniform_double(&state) * cam.aperture; // origin perturbation
                r += radiance(spheres, cubes, planes, beziers, meshes, Ray(p_origin, (hit - p_origin).normalize()),
                              &state) *
                     (1. / samps);
            }
            out_buffer._array[(cam.h - v - 1) * cam.w + u] += r.clamp(0, 1) * .25;
        }
    }
}

void
render_wrapper(const dim3 &nblocks, const dim3 &nthreads, KernelArray<Sphere_GPU> spheres, KernelArray<Cube_GPU> cubes,
               KernelArray<Plane_GPU> planes,
               KernelArray<RotaryBezier_GPU> beziers, KernelArray<TriangleMeshObject_GPU> meshes, Camera cam,
               int samps,
               KernelArray<utils::Vector3> out_buffer) {
    render_pt << < nblocks, nthreads >> >
                            (spheres, cubes, planes, beziers, meshes, cam, samps, out_buffer);
}
