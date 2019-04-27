#include <stdio.h>
#include <stdlib.h>
#include "utils/imghelper.h"
#include "utils/scene.hpp"
#include "utils/camera.hpp"

double clamp(double r) {
    return r < 0 ? 0 : r > 1 ? 1 : r;
}

inline int toInt(double x) { return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }

int main(int argc, char **argv) {
    using namespace utils;
    if (argc != 2)
        return 0;
    Scene scene;
    scene.addObject(new Plane(Vector3(-1, 0, 0), 1, Vector3(.75, .25, .25), Vector3(), DIFF, 1.5)); //left
    scene.addObject(new Plane(Vector3(1, 0, 0), 399, Vector3(.25, .25, .75), Vector3(), DIFF, 1.5)); // right
    scene.addObject(new Cube(Vector3(380, 40, 0), Vector3(395, 35, 100), Vector3(.15, .35, .55), Vector3(), DIFF, 1.3));
    scene.addObject(new Plane(Vector3(0, 0, 1), 500, Vector3(.75, .75, .75), Vector3(), DIFF, 1.5)); //front
    scene.addObject(new Sphere(Vector3(150, 1e5, 181.6), 1e5, Vector3(.75, .75, .75), Vector3(), DIFF, 1.5)); //bottom
    scene.addObject(
            new Sphere(Vector3(50, -1e5 + 381.6, 81.6), 1e5, Vector3(.75, .75, .75), Vector3(), DIFF, 1.5)); //top
    scene.addObject(new Sphere(Vector3(373, 16.5, 78), 16.5, Vector3(.9, .9, .5) * .999, Vector3(), REFR, 1.5));
    scene.addObject(new Sphere(Vector3(250, 981.6 - .63, 81.6), 600, Vector3(), Vector3(33, 33, 22), DIFF, 1.5));
    //scene.addObject(new Cube(Vector3(267, 30, 167), Vector3(327, 30.5, 227), Vector3(.75, .75, .75), Vector3(), DIFF, 1.5));

    cv::Mat _grunge = cv::imread("../texture/b&w_grunge.png");
    cv::Mat _watercolor = cv::imread("../texture/watercolor.jpg");
    Texture grunge_texture(Vector3(.75, .75, .75), Vector3(), DIFF, 1.5);
    Texture watercolor_texture(Vector3(.9, .9, .5) * .999, Vector3(), DIFF, 1.5);
    grunge_texture.pt.mapped_image = cvMat2Arr(_grunge);
    grunge_texture.pt.mapped_transform = utils::Transform2D(2, 0, 0, 2);
    watercolor_texture.pt.mapped_image = cvMat2Arr(_watercolor);
    watercolor_texture.pt.mapped_transform = utils::Transform2D(1/M_PI, 0, 0, .5/M_PI, 0, 0.25);
    scene.addObject(new Plane(Vector3(0, 0, -1), 0, grunge_texture)); // back
    scene.addObject(new Sphere(Vector3(327, 20, 97), 20, watercolor_texture));

    // bezier part
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
    watercolor_texture.pt.mapped_transform = utils::Transform2D(-1., 0, 0, .5/M_PI, 0, 0.25);
    scene.addObject(new RotaryBezier(Vector3(297, 3, 197), ctrl_pnts, watercolor_texture));
    int w = 1920, h = 1080;
    Camera cam(w, h);
    cam.setPosition(Vector3(150, 30, 295.6), Vector3(0.35, -0.030612, -0.4).normalize());
    cam.setLensParam(0.5135, 0., 310);
    cam.setScene(&scene);
    auto res = cam.renderPt(atoi(argv[1]) / 4);
    FILE *f = fopen("image.ppm", "w");
    fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
    for (int i = 0; i < w * h; ++i)
        fprintf(f, "%d %d %d ", toInt(res[i].x()), toInt(res[i].y()), toInt(res[i].z()));
    fclose(f);
    return 0;
}
