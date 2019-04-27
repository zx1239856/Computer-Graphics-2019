/*
 * A simple file to test data transmission between Nvidia GPU and CPU
 * Based on Thrust
 */

#include "cuda_helpers.h"
#include <iostream>
#include <vector>
#include <thrust/host_vector.h>
#include "../common/geometry.hpp"
#include "../utils/bezier.hpp"
#include "bezier.hpp"
#include "object.hpp"

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

using namespace utils;

__global__ void test_kernel(utils::Point2D *output) {
    utils::Point2D p1(1.23456, 9.999999), p2(0, 2);
    *output = utils::Point2D(p1.distance(utils::Point2D(0, 0)), p2.distance(p1));
}

__global__ void calcBezier(KernelArray<Point2D> ctrl, KernelArray<Point2D> coeff, KernelArray<Point2D> slices,
                           KernelArray<double> slices_param, KernelArray<Point2D> result) {
    Bezier2D_GPU bezier;
    bezier._ctrl_pnts = ctrl, bezier._coeff = coeff, bezier._slices = slices, bezier._slices_param = slices_param;
    for (size_t i = 0; i < result._size; ++i) {
        result._array[i] = bezier.getPoint(static_cast<double>(i) / result._size);
    }
}

int main() {
    // device property query
    printDeviceProperty();

    // kernel call test
    utils::Point2D *gpu_pnt;
    utils::Point2D *cpu_pnt = new utils::Point2D();
    CUDA_SAFE_CALL(cudaMalloc(&gpu_pnt, sizeof(utils::Point2D)));
    test_kernel << < 1, 1 >> > (gpu_pnt);
    cudaDeviceSynchronize();
    CUDA_SAFE_CALL(cudaMemcpy(cpu_pnt, gpu_pnt, sizeof(utils::Point2D), cudaMemcpyDeviceToHost));
    cudaFree(gpu_pnt);
    std::cout << "GPU CALC POINT IS: (" << cpu_pnt->x << ", " << cpu_pnt->y << ")" << std::endl;
    delete cpu_pnt;

    // bezier test
    std::vector<utils::Point2D> ctrl_pnts = {{260., 0.},
                                             {300., 50.},
                                             {200., 210.},
                                             {248., 400.},
                                             {350., 500.},
                                             {320., 600.},
                                             {300., 700.},
                                             {300., 800.},
                                             {250., 800.}};
    utils::Bezier2D bezier(ctrl_pnts);
    auto &&coeff = bezier.getAllCoeffs();
    auto &&slices = bezier.getAllSlices();
    auto &&slicesParam = bezier.getAllSlicesParam();
    thrust::device_vector<Point2D> _ctrl(ctrl_pnts.begin(), ctrl_pnts.end());
    thrust::device_vector<Point2D> _coeff(coeff.begin(), coeff.end());
    thrust::device_vector<Point2D> _slices(slices.begin(), slices.end());
    thrust::device_vector<double> _slicesParam(slicesParam.begin(), slicesParam.end());
    thrust::device_vector<Point2D> result(1000);
    calcBezier << < 1, 1 >> >
                       (convertToKernel(_ctrl), convertToKernel(_coeff), convertToKernel(_slices), convertToKernel(
                               _slicesParam),
                               convertToKernel(result));
    cudaDeviceSynchronize();
    thrust::host_vector<Point2D> rr = result;
    cv::Mat image = cv::Mat(1000, 1000, CV_8UC3, cv::Scalar(0, 0, 0));
    for (auto &x:rr) {
        setPixel(image, cvRound(x.x), cvRound(x.y), 255, 255, 0);
    }
    cv::imshow("Result", image);
    cv::waitKey(0);
    return 0;
}