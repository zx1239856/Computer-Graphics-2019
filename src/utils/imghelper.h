//
// Created by zx on 19-4-26.
//

#ifndef HW2_IMGHELPER_H
#define HW2_IMGHELPER_H

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include "../common/geometry.hpp"
#include "../common/common.h"

inline utils::Vector3 getPixel(const cv::Mat &img, int x, int y) {
    if (img.type() != CV_8UC3) {
        printf("Image type incorrect, expected 8UC3\n");
        exit(EXIT_FAILURE);
    }
    return utils::Vector3(img.at<cv::Vec3b>(y, x)[2], img.at<cv::Vec3b>(y, x)[1], img.at<cv::Vec3b>(y, x)[0]);
}

inline triplet<uchar, uchar, uchar> getPixel(const cv::Mat &img, int x, int y, bool unused) {
    auto &&v = img.at<cv::Vec3b>(y,x);
    return {v[2], v[1], v[0]};
}

inline std::vector<utils::Vector3> cvMat2FlatArr(const cv::Mat &img) {
    std::vector<utils::Vector3> res(img.rows * img.cols);
    for (size_t i = 0; i < img.rows; ++i)
        for (size_t j = 0; j < img.cols; ++j)
            res[i * img.cols + j] = getPixel(img, j, i);
    return res;
}

inline std::vector<std::vector<utils::Vector3>> cvMat2Arr(const cv::Mat &img) {
    std::vector<std::vector<utils::Vector3>> res(img.rows);
    for (size_t i = 0; i < img.rows; ++i) {
        res[i].resize(img.cols);
        for (size_t j = 0; j < img.cols; ++j)
            res[i][j] = getPixel(img, j, i);
    }
    return res;
}

#ifdef __NVCC__
cudaTextureObject_t cvMat2CudaTexture(const cv::Mat &img) {
    cudaResourceDesc res_desc;
    uchar4 *buffer;
    size_t size = img.cols * img.rows * sizeof(uchar4);
    CUDA_SAFE_CALL(cudaMalloc(&buffer, size));
    uchar4 *host_mem = new uchar4[img.cols * img.rows];
    for(int i = 0; i < img.rows; ++i)
    {
        for(int j = 0; j < img.cols; ++j)
        {
            auto &&v = getPixel(img, j, i, true);
            int pos = i * img.cols + j;
            host_mem[pos].x = v.first; host_mem[pos].y = v.second; host_mem[pos].z = v.third;
            host_mem[pos].w = 0;
        }
    }
    CUDA_SAFE_CALL(cudaMemcpy(buffer, host_mem, size, cudaMemcpyHostToDevice));
    delete[] host_mem;

    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeLinear;
    res_desc.res.linear.devPtr = buffer;
    res_desc.res.linear.sizeInBytes = size;
    res_desc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
    res_desc.res.linear.desc.x = 8;
    res_desc.res.linear.desc.y = 8;
    res_desc.res.linear.desc.z = 8;
    res_desc.res.linear.desc.w = 8;

    cudaTextureDesc td;
    memset(&td, 0, sizeof(td));
    td.normalizedCoords = 0;
    td.addressMode[0] = cudaAddressModeWrap;
    td.addressMode[1] = cudaAddressModeWrap;
    td.addressMode[2] = cudaAddressModeWrap;
    td.readMode = cudaReadModeElementType;
    td.sRGB = 0;

    cudaTextureObject_t texture = 0;
    cudaCreateTextureObject(&texture, &res_desc, &td, nullptr);

    return texture;
}

#endif


#endif //HW2_IMGHELPER_H
