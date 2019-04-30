#include <cuda_runtime.h>
#include "cuda_helpers.h"
#include "math_helpers.h"
#include "bezier.hpp"
#include "object.hpp"
#include "render_helpers.h"
#include "../utils/bezier.hpp"
#include <cuda_profiler_api.h>

__global__ void
debug_kernel(KernelArray<Sphere_GPU> spheres, KernelArray<Cube_GPU> cubes, KernelArray<Plane_GPU> planes,
             KernelArray<RotaryBezier_GPU> beziers) {
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
}

__device__ static utils::Vector3
radiance(KernelArray<Sphere_GPU> &spheres, KernelArray<Cube_GPU> &cubes, KernelArray<Plane_GPU> &planes,
         KernelArray<RotaryBezier_GPU> &beziers, const Ray &ray, curandState *state) {
    Ray r = ray;
    utils::Vector3 L, F(1., 1., 1.);
    size_t depth = 0;
    while (true) {
        auto &&res = findFirstIntersect(planes, cubes, spheres, beziers,
                                        r);  // triplet<utils::Vector3, double, pair<utils::Point2D, Texture_GPU*>>
        if (res.second >= INF || res.second < 0)
            return L;
        auto &&prop = res.third.second->pt.getColor(res.third.first, state); // color, refl_t
        L += F.mult(res.third.second->pt.emission);
        F = F.mult(prop.first);
        double p = prop.first.max();
        if (++depth > 5) {
            if (curand_uniform_double(state) < p)
                F /= p;
            else return L;
        }
        bool into = false;
        utils::Vector3 x = r.getVector(res.second), nl =
                res.first.dot(r.direction) < 0 ? into = true, res.first : -res.first;
        Ray reflray = Ray(x, r.direction.reflect(nl));

        switch (prop.second) {
            case REFR: {
                utils::Vector3 d = r.direction.refract(res.first, into ? 1 : res.third.second->pt.re_idx,
                                                into ? res.third.second->pt.re_idx : 1);
                if (d.len2() < EPSILON) // total internal reflection
                {
                    r = reflray;
                    continue;
                }
                double a = res.third.second->pt.re_idx - 1, b = res.third.second->pt.re_idx + 1, R0 = a * a / (b * b), c =
                        1 - (into ? -r.direction.dot(nl) : d.dot(res.first));
                double Re = R0 + (1 - R0) * c * c * c * c * c, Tr = 1 - Re, P = .25 + .5 * Re, RP = Re / P, TP =
                        Tr / (1 - P);
                if (curand_uniform_double(state) < P) {
                    F *= RP;
                    r = reflray;
                    continue;
                }
                else {
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
                const utils::Vector3 u = (abs(nl.x()) > 0.1 ? utils::Vector3(0.0, 1.0, 0.0) : utils::Vector3(1.0, 0.0, 0.0)).cross(nl).normalize();
                const utils::Vector3 v = nl.cross(u);
                const utils::Vector3 sample_d = cosWeightedSampleOnHemisphere(curand_uniform_double(state), curand_uniform_double(state));
                const utils::Vector3 d = (u * sample_d.x() + v * sample_d.y() + nl * sample_d.z()).normalize();
                r = Ray(x, d);
                break;
            }
        }
    }
}

__global__ void render_pt(KernelArray<Sphere_GPU> spheres, KernelArray<Cube_GPU> cubes, KernelArray<Plane_GPU> planes,
                          KernelArray<RotaryBezier_GPU> beziers, Camera cam, int samps,
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
                utils::Vector3 p_origin = cam.origin +
                                          (utils::Vector3(curand_uniform_double(&state) * 1.01,
                                                          curand_uniform_double(&state)) - .5) * 2 *
                                          cam.aperture; // origin perturbation
                r += radiance(spheres, cubes, planes, beziers, Ray(p_origin, (hit - p_origin).normalize()), &state) *
                        (1. / samps);
            }
            out_buffer._array[(cam.h - v - 1) * cam.w + u] += r.clamp(0, 1) * .25;
        }
    }
}

int main(int argc, char **argv) {
    using namespace utils;
    if (argc != 4)
        return 0;
    std::vector<Sphere_GPU> spheres_;
    spheres_.emplace_back(Sphere_GPU(Vector3(150, 1e5, 181.6), 1e5, Vector3(.75, .75, .75), Vector3(), DIFF, 1.5));
    spheres_.emplace_back(
            Sphere_GPU(Vector3(50, -1e5 + 381.6, 81.6), 1e5, Vector3(.75, .75, .75), Vector3(), DIFF, 1.5)); // top
    spheres_.emplace_back(Sphere_GPU(Vector3(375, 16.5 + 8, 25), 16.5, Vector3(.9, .9, .75) * .999, Vector3(), REFR, 1.5));
    spheres_.emplace_back(Sphere_GPU(Vector3(250, 1181.6 - .9, 81.6), 800, Vector3(), Vector3(50, 50, 50), DIFF, 1.5)); // top light
    std::vector<Cube_GPU> cubes_;
    Texture_GPU lightcube;
    lightcube.pt.re_idx = 1.3, lightcube.pt.color = Vector3(0.85, 0.85, 0.7), lightcube.pt.emission = Vector3(),
    lightcube.pt.refl_1 = DIFF, lightcube.pt.refl_2 = REFR, lightcube.pt.probability = 0;
    cubes_.emplace_back(
            Cube_GPU(Vector3(350, 0, 0), Vector3(400, 8, 50), lightcube));
    std::vector<Plane_GPU> planes_;
    planes_.emplace_back(Plane_GPU(Vector3(-1, 0, 0), 1, Vector3(.75, .75, .75), Vector3(), DIFF, 1.5));  // left
    //planes_.emplace_back(Plane_GPU(Vector3(1, 0, 0), 400, Vector3(.25, .25, .75), Vector3(), DIFF, 1.5)); // right
    planes_.emplace_back(Plane_GPU(Vector3(0, 0, 1), 500, Vector3(.75, .75, .75), Vector3(), DIFF, 1.5));  // front
    planes_.emplace_back(Plane_GPU(Vector3(0, 0, -1), 0, Vector3(.25, .5, .75), Vector3(), DIFF, 1.5)); // back
    std::vector<RotaryBezier_GPU> beziers_;

    cv::Mat _oilpainting = cv::imread("../texture/oil_painting.png");
    cv::Mat _watercolor = cv::imread("../texture/watercolor.jpg");
    auto &&oilpainting_arr = cvMat2FlatArr(_oilpainting);
    auto &&watercolor_arr = cvMat2FlatArr(_watercolor);
    Texture_GPU oil_painting, watercolor_texture;
    oil_painting.pt.color = Vector3(.75, .75, .75);
    oil_painting.pt.emission = Vector3();
    oil_painting.pt.re_idx = 1.5;
    oil_painting.pt.refl_1 = DIFF;
    oil_painting.pt.probability = 0;
    oil_painting.pt.img_w = _oilpainting.cols;
    oil_painting.pt.img_h = _oilpainting.rows;
    oil_painting.pt.mapped_image = makeKernelArr(oilpainting_arr);
    oil_painting.pt.mapped_transform = Transform2D(0, -2/450. , 2/600., 0, 2, 0);
    planes_.emplace_back(Plane_GPU(Vector3(1, 0, 0), 400, oil_painting));
    watercolor_texture.pt.color = Vector3(.9, .9, .5) * .999;
    watercolor_texture.pt.emission = Vector3();
    watercolor_texture.pt.re_idx = 1.5;
    watercolor_texture.pt.refl_1 = DIFF;
    watercolor_texture.pt.probability = 0;
    watercolor_texture.pt.img_w = _watercolor.cols;
    watercolor_texture.pt.img_h = _watercolor.rows;
    watercolor_texture.pt.mapped_image = makeKernelArr(watercolor_arr);
    watercolor_texture.pt.mapped_transform = Transform2D(1 / M_PI, 0, 0, .5 / M_PI, 0, 0.25);
    spheres_.emplace_back(Sphere_GPU(Vector3(280, 13, 103), 13, watercolor_texture));

    double xscale = 2, yscale = 2;
    std::vector<Point2D> ctrl_pnts = {{0. / xscale, 0. / yscale},
                                      {13. / xscale, 0. / yscale},
                                      {30. / xscale, 10. / yscale},
                                      {30. / xscale, 20. / yscale},
                                      {30. / xscale, 30. / yscale},
                                      {25. / xscale, 40. / yscale},
                                      {15. / xscale, 50. / yscale},
                                      {10. / xscale, 70. / yscale},
                                      {20. / xscale, 80. / yscale}};
    Bezier2D bezier__(ctrl_pnts);
    auto &&coeff = bezier__.getAllCoeffs();
    auto &&slices = bezier__.getAllSlices();
    auto &&slicesParam = bezier__.getAllSlicesParam();
    Bezier2D_GPU bezier;
    bezier._ctrl_pnts = makeKernelArr(ctrl_pnts), bezier._coeff = makeKernelArr(coeff),
    bezier._slices = makeKernelArr(slices), bezier._slices_param = makeKernelArr(slicesParam);
    bezier.xmax = bezier__.xMax(), bezier.xmin = bezier__.xMin(), bezier.ymax = bezier__.yMax(), bezier.ymin = bezier__.yMin();

    watercolor_texture.pt.mapped_transform = Transform2D(-1., 0, 0, .5 / M_PI, 0, 0.25);
    beziers_.emplace_back(RotaryBezier_GPU(Vector3(297, 3, 197), bezier, watercolor_texture));

    //debug_kernel<<<1,1>>>(convertToKernel(spheres), convertToKernel(cubes), convertToKernel(planes), convertToKernel(beziers));
    //cudaDeviceSynchronize();

    // camera params
    Camera cam = {
            atoi(argv[2]), atoi(argv[3]),
            Vector3(150, 33, 295.6), Vector3(0.4, -0.030612, -0.35).normalize(),
            0.5135, 0., 310
    };

    // render
    const dim3 nblocks(cam.w / 16u, cam.h / 16u);
    const dim3 nthreads(16u, 16u);
    KernelArray<utils::Vector3> gpu_out = createKernelArr<utils::Vector3>(static_cast<size_t>(cam.w * cam.h));
    render_pt << < nblocks, nthreads >> >
                            (makeKernelArr(spheres_), makeKernelArr(cubes_), makeKernelArr(planes_), makeKernelArr(
                                    beziers_), cam, atoi(argv[1])/4, gpu_out);
    std::vector<Vector3> res = makeStdVector(gpu_out);
    releaseKernelArr(oil_painting.pt.mapped_image);
    releaseKernelArr(watercolor_texture.pt.mapped_image);
    releaseKernelArr(bezier._ctrl_pnts); releaseKernelArr(bezier._coeff); releaseKernelArr(bezier._slices); releaseKernelArr(bezier._slices_param);
    releaseKernelArr(gpu_out);
    FILE *f = fopen("image.ppm", "w");
    fprintf(f, "P3\n%d %d\n%d\n", cam.w, cam.h, 255);
    for (int i = 0; i < cam.w * cam.h; ++i)
        fprintf(f, "%d %d %d ", toUInt8(res[i].x()), toUInt8(res[i].y()), toUInt8(res[i].z()));
    fclose(f);
    return 0;
}