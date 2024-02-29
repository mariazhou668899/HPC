/* 1. Complile methods: running using UW GPU computer
 * Step1: navigate to the file holding current file
 * Step3: input nvcc -o GPU_infor Part1_getGPUInfor.cu
 * Step4: input ./GPU_infor
 *
*/

/* 2. Introduction: This program is to get GPU hardware specification by using CUDA APIs.
 *
*/


#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount;

    // Get the number of available CUDA devices
    cudaGetDeviceCount(&deviceCount);

    // Loop through each CUDA device
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;

        // Get the properties of the current CUDA device
        cudaGetDeviceProperties(&prop, i);

        // Print information about the current CUDA device
        std::cout << "GPU " << i << " - " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Multiprocessor Count: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Warp Size: " << prop.warpSize << " threads" << std::endl;
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Threads per Multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Max Grid Dimensions: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;
        std::cout << "  Max Threads Dimensions: (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" <<           std::endl;
        std::cout << "  Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        // Add more properties as needed

    }

    return 0;
}
