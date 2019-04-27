//
// Created by zx on 19-4-26.
//

#ifndef HW2_CUDA_HELPERS_H
#define HW2_CUDA_HELPERS_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <iostream>

void printDeviceProperty(int dev = 0)
{
    cudaDeviceProp devProp;
    if(cudaGetDeviceProperties(&devProp, dev) == cudaSuccess)
    {
        std::cout << "Device " << dev << ", named: " << devProp.name << std::endl;
        std::cout << "Multi Processor Count: " << devProp.multiProcessorCount << std::endl;
        std::cout << "Size of SharedMem Per-Block: " << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
        std::cout << "Max Threads Per-Block: " << devProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max Threads Per-MultiProcessor: " << devProp.maxThreadsPerMultiProcessor << std::endl;
    }
}

inline void handleCudaError(cudaError_t err, const char* file, int line) {
    if (cudaSuccess != err) {
        std::cout << cudaGetErrorString(err) << " in " << file << " at line " << line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_SAFE_CALL(err) (handleCudaError( err, __FILE__, __LINE__ ))

// thrust_vector to kernel
template <typename T>
struct KernelArray
{
    T*  _array;
    int _size;
};

template <typename T>
KernelArray<T> convertToKernel(thrust::device_vector<T>& dVec)
{
    KernelArray<T> kArray;
    kArray._array = thrust::raw_pointer_cast(&dVec[0]);
    kArray._size  = (int) dVec.size();

    return kArray;
}

template <typename T>
KernelArray<T> makeKernelArr(std::vector<T> &src)
{
    KernelArray<T> kArray;
    CUDA_SAFE_CALL(cudaMalloc((void**)&kArray._array, sizeof(T) * src.size()));
    CUDA_SAFE_CALL(cudaMemcpy(kArray._array, src.data(), sizeof(T) * src.size(), cudaMemcpyHostToDevice));
    kArray._size = src.size();
    return kArray;
}

template <typename T>
KernelArray<T> createKernelArr(size_t size)
{
    KernelArray<T> kArray;
    CUDA_SAFE_CALL(cudaMalloc((void**)&kArray._array, sizeof(T) * size));
    kArray._size = size;
    return kArray;
}

template <typename T>
std::vector<T> makeStdVector(KernelArray<T> arr) {
    std::vector<T> res(arr._size);
    CUDA_SAFE_CALL(cudaMemcpy(res.data(), arr._array, sizeof(T) * arr._size, cudaMemcpyDeviceToHost));
    return res;
}

template <typename T>
void releaseKernelArr(KernelArray<T> arr) {
    CUDA_SAFE_CALL(cudaFree(arr._array));
}

template <typename T, typename K>
struct pair
{
        T first;
        K second;
};

template <typename T, typename K, typename L>
struct triplet
{
    T first;
    K second;
    L third;
};

#endif //HW2_CUDA_HELPERS_H
