/*
 * Created by zx on 19-5-15.
 */
#include "gpu/render.cuh"

int main(int argc, char **argv) {
    using namespace utils;
    if (argc != 4)
        return 0;
    printDeviceProperty();
    std::vector<Sphere_GPU> spheres_;
    std::vector<Cube_GPU> cubes_;
    std::vector<Plane_GPU> planes_;
    std::vector<RotaryBezier_GPU> beziers_;
    std::vector<TriangleMeshObject_GPU> meshes_;
    spheres_.emplace_back(
            Sphere_GPU(Vector3(50, -1e5 + 281.6, 81.6), 1e5, Vector3(.75, .75, .75), Vector3(), BRDFs[WALL])); // top
    spheres_.emplace_back(Sphere_GPU(Vector3(350, 1081.6 - 1.3, 231.6), 800, Vector3(), Vector3(60, 60, 60),
                                     BRDFs[LIGHT])); // top light
    Texture_GPU lightcube;
    lightcube.re_idx = 1.3, lightcube.color = Vector3(.15, .35, .55), lightcube.emission = Vector3(),
            lightcube.setBRDF(BRDFs[DIFFUSE]);
    cubes_.emplace_back(
            Cube_GPU(Vector3(340, 0, 0), Vector3(400, 5, 60), lightcube));
    planes_.emplace_back(Plane_GPU(Vector3(-1, 0, 0), 1, Vector3(.75, .75, .75), Vector3(), BRDFs[WALL]));  // left
    //planes_.emplace_back(Plane_GPU(Vector3(1, 0, 0), 400, Vector3(.25, .25, .75), Vector3(), DIFF, 1.5)); // right
    planes_.emplace_back(Plane_GPU(Vector3(0, 0, 1), 500, Vector3(.75, .75, .75), Vector3(), BRDFs[WALL]));  // front
    
    cv::Mat _oilpainting = cv::imread("../texture/oil_painting.png");
    cv::Mat _watercolor = cv::imread("../texture/watercolor.jpg");
    cv::Mat _floor = cv::imread("../texture/floor.jpg");
    cv::Mat _wall = cv::imread("../texture/wall.jpg");
    cudaTextureObject_t oilpainting = cvMat2CudaTexture(_oilpainting);
    cudaTextureObject_t watercolor = cvMat2CudaTexture(_watercolor);
    cudaTextureObject_t floor = cvMat2CudaTexture(_floor);
    cudaTextureObject_t wall = cvMat2CudaTexture(_wall);
    Texture_GPU oil_painting, watercolor_texture, floor_texture, wall_texture;
    floor_texture.color = Vector3(.75, .75, .75);
    floor_texture.emission = Vector3();
    floor_texture.setBRDF(BRDFs[WALL]);
    floor_texture.img_w = _floor.cols;
    floor_texture.img_h = _floor.rows;
    floor_texture.mapped_image = floor;
    floor_texture.mapped_transform = Transform2D(0, -5 / 918., 5 / 1024., 0, 2, 0);
    wall_texture.color = Vector3(.75, .75, .75);
    wall_texture.emission = Vector3();
    wall_texture.setBRDF(BRDFs[WALL]);
    wall_texture.img_w = _wall.cols;
    wall_texture.img_h = _wall.rows;
    wall_texture.mapped_image = wall;
    wall_texture.mapped_transform = Transform2D(0, -2 / 1350., 2 / 2400., 0, 0, 0);
    oil_painting.color = Vector3(.75, .75, .75);
    oil_painting.emission = Vector3();
    oil_painting.setBRDF(BRDFs[WALL]);
    oil_painting.img_w = _oilpainting.cols;
    oil_painting.img_h = _oilpainting.rows;
    oil_painting.mapped_image = oilpainting;
    oil_painting.mapped_transform = Transform2D(0, -2 / 450., 2 / 600., 0, 2, 0);
    planes_.emplace_back(Plane_GPU(Vector3(1, 0, 0), 400, oil_painting));
    watercolor_texture.color = Vector3(.9, .9, .5) * .999;
    watercolor_texture.emission = Vector3();
    watercolor_texture.setBRDF(BRDFs[DIFFUSE]);
    watercolor_texture.img_w = _watercolor.cols;
    watercolor_texture.img_h = _watercolor.rows;
    watercolor_texture.mapped_image = watercolor;
    watercolor_texture.mapped_transform = Transform2D(1 / M_PI, 0, 0, .5 / M_PI, 0, 0.25);
    //spheres_.emplace_back(Sphere_GPU(Vector3(280, 13, 103), 13, watercolor_texture));
    spheres_.emplace_back(Sphere_GPU(Vector3(265, 13, 100), 13, Vector3(.75, .75, .75), Vector3(), BRDFs[METAL]));
    spheres_.emplace_back(Sphere_GPU(Vector3(300, 10, 200), 10, Vector3(.75, .9, .9), Vector3(), BRDFs[GLASS]));
    spheres_.emplace_back(Sphere_GPU(Vector3(280, 8, 135), 8, Vector3(.75, .9, .65), Vector3(), BRDFs[GLASS]));
    spheres_.emplace_back(Sphere_GPU(Vector3(270, 5, 155), 5, Vector3(.75, .75, .35), Vector3(), BRDFs[GLASS]));
    planes_.emplace_back(Plane_GPU(Vector3(0, 0, -1), 0, wall_texture)); // back
    planes_.emplace_back(Plane_GPU(Vector3(0, 1, 0), 0, floor_texture)); // bottom
    double xscale = 1.5, yscale = 1.5;
    std::vector<Point2D> ctrl_pnts = {{0. / xscale,  0. / yscale},
                                      {13. / xscale, 0. / yscale},
                                      {30. / xscale, 10. / yscale},
                                      {30. / xscale, 20. / yscale},
                                      {30. / xscale, 30. / yscale},
                                      {25. / xscale, 40. / yscale},
                                      {15. / xscale, 50. / yscale},
                                      {10. / xscale, 70. / yscale},
                                      {20. / xscale, 80. / yscale}};
    Bezier2D cpu_bezier(ctrl_pnts);

    watercolor_texture.mapped_transform = Transform2D(-1., 0, 0, .5 / M_PI, 0, 0.25);
    beziers_.emplace_back(RotaryBezier_GPU(Vector3(370, 5.5, 30), cpu_bezier.toGPU(), watercolor_texture));

    auto param = loadObject("../model/angel_lucy.obj");
    KDTree cpu_tree(std::get<0>(param), std::get<1>(param), std::get<2>(param));
    KDTree_GPU gpu_tree = cpu_tree.toGPU();
    meshes_.emplace_back(
            TriangleMeshObject_GPU(utils::Vector3(345, .5 - 1.19, 169), 0.1, gpu_tree, Vector3(.75, .75, .75), Vector3(),
                                   BRDFs[DIFFUSE]));

    //debug_kernel<<<1,1>>>(convertToKernel(spheres), convertToKernel(cubes), convertToKernel(planes), convertToKernel(beziers));
    //cudaDeviceSynchronize();

    // camera params
    Camera cam = {
            atoi(argv[2]), atoi(argv[3]),
            Vector3(150, 40, 295.6), Vector3(0.4, -0.008612, -0.35).normalize(),
            0.5535, 3.3, 223
    };

    // render
    const dim3 nblocks(cam.w / 16u, cam.h / 16u);
    const dim3 nthreads(16u, 16u);
    KernelArray<utils::Vector3> gpu_out = createKernelArr<utils::Vector3>(static_cast<size_t>(cam.w * cam.h));
    printf("Memory copied to GPU, now start executing render kernel...\n");
    render_wrapper(nblocks, nthreads, makeKernelArr(spheres_), makeKernelArr(cubes_), makeKernelArr(planes_), makeKernelArr(
                                    beziers_), makeKernelArr(meshes_), cam, atoi(argv[1]) / 4, gpu_out);
    std::vector<Vector3> res = makeStdVector(gpu_out);
    releaseKernelArr(gpu_out);
    FILE *f = fopen("image.ppm", "w");
    fprintf(f, "P3\n%d %d\n%d\n", cam.w, cam.h, 255);
    for (int i = 0; i < cam.w * cam.h; ++i)
        fprintf(f, "%d %d %d ", toUInt8(res[i].x()), toUInt8(res[i].y()), toUInt8(res[i].z()));
    fclose(f);
    return 0;
}
