/*
 * Created by zx on 19-4-3.
 */
#include "bezier.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

void setPixel(cv::Mat &img, int x, int y, uint8_t R, uint8_t G, uint8_t B) {
    if (x < 0 || x >= img.cols)
        return;
    if (y < 0 || y >= img.rows)
        return;
    img.at<cv::Vec3b>(y, x)[0] = B;
    img.at<cv::Vec3b>(y, x)[1] = G;
    img.at<cv::Vec3b>(y, x)[2] = R;
}

int main()
{
    using namespace utils;
    std::vector<Point2D> ctrl_pnts = {{260.,    0.},
    {300.,  50.},
    {200.,  210.},
    {248.,  400.},
    {350.,  500.},
    {320.,  600.},
    {300.,  700.},
    {300.,  800.},
    {250.,  800.}};
    Bezier2D bezier(ctrl_pnts);
    cv::Mat image = cv::Mat(1000, 1000, CV_8UC3, cv::Scalar(0,0,0));
    for(const auto &p: ctrl_pnts)
        setPixel(image, p.first, p.second, 255, 255, 255);
    constexpr size_t sample = 1000;
    for(size_t i = 0; i < sample; ++i)
    {
        auto p = bezier.getPoint(static_cast<double>(i)/sample);
        setPixel(image, cvRound(p.first), cvRound(p.second), 255, 255, 0);
    }
    printf("xmin: %lf, xmax: %lf, ymin: %lf, ymax: %lf\n", bezier.xMin(), bezier.xMax(), bezier.yMin(), bezier.yMax());
    cv::imshow("Result", image);
    cv::waitKey(0);
}
