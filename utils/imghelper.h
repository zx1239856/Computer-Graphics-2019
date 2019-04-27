//
// Created by zx on 19-4-26.
//

#ifndef HW2_IMGHELPER_H
#define HW2_IMGHELPER_H

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include "../common/geometry.hpp"

utils::Vector3 getPixel(const cv::Mat &img, int x, int y) {
    if (img.type() != CV_8UC3) {
        printf("Image type incorrect, expected 8UC3\n");
        exit(EXIT_FAILURE);
    }
    return utils::Vector3(img.at<cv::Vec3b>(y, x)[2], img.at<cv::Vec3b>(y, x)[1], img.at<cv::Vec3b>(y, x)[0]);
}

std::vector<utils::Vector3> cvMat2FlatArr(const cv::Mat &img) {
    std::vector<utils::Vector3> res(img.rows * img.cols);
    for (size_t i = 0; i < img.rows; ++i)
        for (size_t j = 0; j < img.cols; ++j)
            res[i * img.cols + j] = getPixel(img, j, i);
    return res;
}

std::vector<std::vector<utils::Vector3>> cvMat2Arr(const cv::Mat &img) {
    std::vector<std::vector<utils::Vector3>> res(img.rows);
    for (size_t i = 0; i < img.rows; ++i) {
        res[i].resize(img.cols);
        for (size_t j = 0; j < img.cols; ++j)
            res[i][j] = getPixel(img, j, i);
    }
    return res;
}


#endif //HW2_IMGHELPER_H
