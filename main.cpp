#include <stdio.h>
#include <stdlib.h>
#include "utils/scene.hpp"

double clamp(double r)
{
    return r < 0 ? 0 : r > 1 ? 1 : r;
}
inline int toInt(double x){ return int(pow(clamp(x),1/2.2)*255+.5); }

int main(int argc, char **argv) {
    using namespace utils;
    if(argc != 2)
        return 0;
    Scene scene;
    scene.addObject(new Plane(Vector3(-1, 0, 0), 1, Vector3(.75,.25,.25), Vector3(), DIFF, 1.5)); //left
    scene.addObject(new Plane(Vector3(1, 0, 0), 99, Vector3(.25,.25,.75), Vector3(), DIFF, 1.5)); // right
    scene.addObject(new Plane(Vector3(0, 0, 1), 0, Vector3(.25,.75,.25), Vector3(), DIFF, 1.5)); // back
    scene.addObject(new Cube(Vector3(80,40,0),Vector3(95, 35, 100),Vector3(.15,.35,.55), Vector3() , DIFF, 1.3));
    //scene.addObject(new Sphere(Vector3(50,40.8,-1e5+170), 1e5, Vector3(), Vector3(), DIFF, 1.5)); //front
    scene.addObject(new Sphere(Vector3(50, 1e5, 81.6), 1e5, Vector3(.75,.75,.75), Vector3(), DIFF, 1.5)); //bottom
    scene.addObject(new Sphere(Vector3(50,-1e5+81.6,81.6), 1e5, Vector3(.75,.75,.75), Vector3(), DIFF, 1.5)); //top
    scene.addObject(new Sphere(Vector3(27,16.5,47), 16.5, Vector3(1, 1, 1) * .999, Vector3(), REFR, 1.5));
    scene.addObject(new Sphere(Vector3(73,16.5,78), 16.5, Vector3(.9,.9,.5)*.999, Vector3(), REFR, 1.5));
    scene.addObject(new Sphere(Vector3(10, 0, 0), 10, Vector3(.75,.75,.15), Vector3(), DIFF, 1.5));
    scene.addObject(new Sphere(Vector3(50,681.6-.27,81.6), 600, Vector3(), Vector3(12,12,12), DIFF, 1.5));
    Camera cam(Vector3(50, 60, 395.6), Vector3(0, -0.042612, -0.4).normalize());
    int w= 1920, h = 1080;
    auto res = scene.render(cam, atoi(argv[1])/4 , w, h);
    FILE *f = fopen("image.ppm", "w");
    fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
    for(int i = 0; i < w*h; ++i)
        fprintf(f, "%d %d %d ", toInt(res[i].x()), toInt(res[i].y()), toInt(res[i].z()));
    fclose(f);
    return 0;
}