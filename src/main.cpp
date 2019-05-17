#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "utils/imghelper.h"
#include "utils/scene.hpp"
#include "utils/camera.hpp"
#include "obj/obj_wrapper.h"

void printUsage(char *name) {
	printf("%s [ALGORITHM] [PARAMS]\n"
		   "Format:\n PT [width] [height] [spp]\n"
		   "SPPM [num_rounds] [width] [height] [num_rounds] [num_photons_per_round]\n", name);
}

inline int toInt(double x) { return int(pow(clampVal(x), 1 / 2.2) * 255 + .5); }
//inline int toInt(double x) { return int(clampVal(pow(1-exp(-x),1/2.2))*255+.5); }

int main(int argc, char **argv) {
    using namespace utils;
    if (argc != 5 && argc != 6) {
    	printUsage(argv[0]);
        return 0;
    }
    std::string alg(argv[1]);
    Scene scene;
    scene.addObject(new Plane(Vector3(-1, 0, 0), 1, Vector3(.75, .25, .25), Vector3(), BRDFs[WALL])); //left
    scene.addObject(new Plane(Vector3(1, 0, 0), 399, Vector3(.25, .25, .75), Vector3(), BRDFs[WALL])); // right
    scene.addObject(new Cube(Vector3(340, 0, 0), Vector3(400, 5, 60), Vector3(.15, .35, .55), Vector3(), BRDFs[DIFFUSE]));
    scene.addObject(new Plane(Vector3(0, 0, 1), 500, Vector3(.75, .75, .75), Vector3(), BRDFs[WALL])); //front
    scene.addObject(new Sphere(Vector3(150, 1e5, 181.6), 1e5, Vector3(.75, .75, .75), Vector3(), BRDFs[FLOOR])); //bottom
    scene.addObject(
            new Sphere(Vector3(50, -1e5 + 381.6, 81.6), 1e5, Vector3(.75, .75, .75), Vector3(), BRDFs[WALL])); //top
    //scene.addObject(new Sphere(Vector3(373, 16.5, 78), 16.5, Vector3(.9, .9, .5) * .999, Vector3(), BRDFs[GLASS]));
    scene.addObject(new Sphere(Vector3(300, 981.6 - .63, 183), 600, Vector3(), Vector3(33, 33, 22), BRDFs[LIGHT])); // light
    //scene.addObject(new Cube(Vector3(267, 30, 167), Vector3(327, 30.5, 227), Vector3(.75, .75, .75), Vector3(), DIFF, 1.5));
    scene.setLight(Vector3(300, 381.6 - 1, 183), 30, Vector3(0, -1, 0));

    cv::Mat _oilpainting = cv::imread("../texture/oil_painting.png");
    Texture oilpainting_texture(Vector3(.9, .9, .5) * .999, Vector3(), BRDFs[DIFFUSE]);
    oilpainting_texture.mapped_image = cvMat2Arr(_oilpainting);
    oilpainting_texture.mapped_transform = utils::Transform2D(0, -2 / 450., 2 / 600., 0, 0, 0.5);
    scene.addObject(new Plane(Vector3(0, 0, -1), 0, oilpainting_texture)); // back
    //scene.addObject(new Sphere(Vector3(327, 20, 97), 20, watercolor_texture));

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
    oilpainting_texture.mapped_transform = utils::Transform2D(-1., 0, 0, .5/M_PI, 0, 0.25);
    //scene.addObject(new RotaryBezier(Vector3(370, 5.5, 30), ctrl_pnts, Vector3(.75, .75, .75), Vector3(), BRDFs[DIFFUSE]));
    auto param = loadObject("../model/angel_lucy.obj");
    scene.addObject(new TriangleMeshObject(utils::Vector3(330, 1-1.19, 160), 0.1, std::get<0>(param), std::get<1>(param), std::get<2>(param), Vector3(.75, .75, .75), Vector3(), BRDFs[DIFFUSE]));
    //scene.addObject(new Sphere(Vector3(320, 10.1, 153), 10, Vector3(.9, .9, .9), Vector3(), BRDFs[METAL]));
    scene.addObject(new Sphere(Vector3(270, 15.1, 100), 15, Vector3(.75, .9, .9), Vector3(), BRDFs[GLASS]));
    scene.addObject(new Sphere(Vector3(270, 5.1, 120), 5, Vector3(.9, .9, .75), Vector3(), BRDFs[GLASS]));
    scene.addObject(new Sphere(Vector3(250, 5.1, 130), 5, Vector3(.9, .75, .9), Vector3(), BRDFs[GLASS]));
	int w = atoi(argv[2]), h = atoi(argv[3]);
    Camera cam(w, h);
    cam.setPosition(Vector3(150, 50, 295.6), Vector3(0.35, -0.030612, -0.4).normalize());
    cam.setLensParam(0.5135, 0., 310);
    cam.setScene(&scene);
    std::vector<utils::Vector3> res;
    if(alg == "PT") {
    	res = cam.renderPt(atoi(argv[4]) / 4);
    } else if(alg == "SPPM") {
    	res = cam.renderPPM(atoi(argv[4]), atoi(argv[5]));
    } else {
    	printf("Unsupported algorithm %s detected\n", argv[1]);
    	printUsage(argv[0]);
    }
    FILE *f = fopen("image.ppm", "w");
    fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
    for (int i = 0; i < w * h; ++i)
        fprintf(f, "%d %d %d ", toInt(res[i].x()), toInt(res[i].y()), toInt(res[i].z()));
    fclose(f);
    return 0;
}
