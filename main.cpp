#include <stdio.h>
#include "utils/scene.hpp"

double clamp(double r)
{
    return r < 0 ? 0 : r > 1 ? 1 : r;
}
inline int toInt(double x){ return int(pow(clamp(x),1/2.2)*255+.5); }

int main() {
    using namespace utils;
    Scene scene;
    scene.addObject(new Sphere(Vector3(1e5+1, 40.8, 81.6), 1e5, Vector3(.75,.25,.25), Vector3(), DIFF, 1.5));
    scene.addObject(new Sphere(Vector3(-1e5+99, 40.8, 81.6), 1e5, Vector3(.25,.25,.75), Vector3(), DIFF, 1.5));
    scene.addObject(new Sphere(Vector3(50, 40.8, 1e5), 1e5, Vector3(.75,.75,.75), Vector3(), DIFF, 1.5));
    scene.addObject(new Sphere(Vector3(50,40.8,-1e5+170), 1e5, Vector3(), Vector3(), DIFF, 1.5));
    scene.addObject(new Sphere(Vector3(50, 1e5, 81.6), 1e5, Vector3(.75,.75,.75), Vector3(), DIFF, 1.5));
    scene.addObject(new Sphere(Vector3(50,-1e5+81.6,81.6), 1e5, Vector3(.75,.75,.75), Vector3(), DIFF, 1.5));
    scene.addObject(new Sphere(Vector3(27,16.5,47), 16.5, Vector3(1, 1, 1) * .999, Vector3(), SPEC, 1.5));
    scene.addObject(new Sphere(Vector3(73,16.5,78), 16.5, Vector3(1, 1, 1) * .999, Vector3(), REFR, 1.5));
    scene.addObject(new Sphere(Vector3(50,681.6-.27,81.6), 600, Vector3(), Vector3(12,12,12), DIFF, 1.5));
    Camera cam(Vector3(50, 52, 295.6), Vector3(0, -0.042612, -1).normalize());
    int w= 1920, h = 1080;
    auto res = scene.render(cam, 1300, w, h);
    FILE *f = fopen("image.ppm", "w");
    fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
    for(int i = 0; i < w*h; ++i)
        fprintf(f, "%d %d %d ", toInt(res[i].x()), toInt(res[i].y()), toInt(res[i].z()));
    fclose(f);
    return 0;
}