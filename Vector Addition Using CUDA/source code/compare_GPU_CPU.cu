/* 1. Complile methods: running using UW GPU computer
 * Step1: Login the GPU computer
 * Step1: navigate to the file holding current file
 * Step3: input nvcc -o gpu_cpu compare_GPU_CPU.cu
 * Step4: input ./gpu_cpu
 *
*/

/* 2. In this program, it will conduct experiment 5: 
  1) Experiment 5: GPU_CPU_compae: #Blocks = 1024, #Threads = 1024
  std::vector<size_t>size2 = {10,13,16,17};
  std::ofstream outputFile21("experiment_GPU_op.txt");
  std::ofstream outputFile22("experiment_GPU_m.txt");
 * 
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cmath>

// Function to generate vector with random integer between -10 and 10
void generateRandomVector(std::vector<int>& vec, size_t size) {
    vec.resize(size);
    for (size_t i = 0; i < size; ++i) {
        // Generate random integer between -10 and 10
        vec[i] = rand() % 21 - 10;
    }
}


// Function to realize a kernel for vector addition
__global__ void vectorAdditionKernel(const int* a, const int* b, int* result, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

// Function to print a vector
int printVector(const std::vector<int>& vec, const char* name) {
    std::cout << name << std::endl;
    std::cout << "{" << std::endl;

    for (size_t i = 0; i < vec.size(); ++i) {
        int element = vec[i];
        std::cout << "\t" << element;
    }

    std::cout << std::endl << "}" << std::endl;

    return 0;
}

// Function to perform 1D vector addition using CUDA and measure time
float vectorAdditionCUDA(const std::vector<int>& vectorA, const std::vector<int>& vectorB, std::vector<int>& resultVector,
                         size_t numBlocks, size_t numThreadsPerBlock) {
    size_t vectorSize = vectorA.size();
    // Allocate memory on the device
    int* d_a, *d_b, *d_result;
    cudaMalloc(&d_a, vectorSize * sizeof(int));
    cudaMalloc(&d_b, vectorSize * sizeof(int));
    cudaMalloc(&d_result, vectorSize * sizeof(int));

    // Copy input vectors to device
    cudaMemcpy(d_a, vectorA.data(), vectorSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, vectorB.data(), vectorSize * sizeof(int), cudaMemcpyHostToDevice);
    // Configure execution parameters
    dim3 blockSize(numThreadsPerBlock);
    dim3 gridSize(numBlocks);

    // Start measuring time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // Launch kernel
    vectorAdditionKernel<<<gridSize, blockSize>>>(d_a, d_b, d_result, vectorSize);

    // Stop measuring time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    // Calculate elapsed time
    float milliseconds = 0;
    auto elapsedTime = cudaEventElapsedTime(&milliseconds, start, stop);
    // Copy result back to host
    cudaMemcpy(resultVector.data(), d_result, vectorSize * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    // Return elapsed time in milliseconds
    return milliseconds;
}

int main() {
    // Seed for reproducibility
    srand(static_cast<unsigned>(time(nullptr)));


    // Experiment 4: configurations. Varying #block, #Threads 
    std::vector<size_t> numBlocksList4 = {1024};

    std::ofstream outputFile21("experiment_GPU_op.txt");
    std::ofstream outputFile22("experiment_GPU_m.txt");

 
  	
  	// Experiment GPU_CPU_compae: #Blocks = 1024, #Threads = 1024
  	// size2 = {3 - 20};
  	for (int i = 3; i <= 20; i++) {
  		
      // Calculate vectorsize
  		size_t vectorSize = 1<<i;
          
          // Creat and initialize vectorA, vectorB and resultVector
          std::vector<int> vectorA, vectorB;
          generateRandomVector(vectorA, vectorSize);
          generateRandomVector(vectorB, vectorSize);
          std::vector<int> resultVector(vectorSize, 0);
  
          if (vectorSize == 16) {
            // Print vectorA before kernel execution
            printVector(vectorA, "Vector A Before experiments 5, size == 16: ");
            
            // Print vectorB before kernel execution
            printVector(vectorB, "Vector B Before experiments 5, size == 16: ");
            
            // Print resultVector before kernel execution
            printVector(resultVector, "resultVector Before experiments 5, size == 16: ");
            std::cout << std::endl;
          }
          
          // numBlocksList2 = {1, 8,64,128};  
          for (size_t j = 0; j < numBlocksList4.size(); j++) {
              
              // Get the block number
  			      size_t numBlocks = numBlocksList4[j];
  			      
              // Record the start time including memory and operation 
              auto startTime2 = std::chrono::high_resolution_clock::now();
              
              // Calculate CUDA vector addition and return the adding operation time
              float elapsedTime2 = vectorAdditionCUDA(vectorA, vectorB, resultVector, numBlocks, 1024);
              
              // Record the end time just including memory
              auto endTime2 = std::chrono::high_resolution_clock::now();
              
              // Calculate CUDA vector addtion time including memory and adding operation
              std::chrono::duration<double, std::milli> elapsedMilliseconds2 = endTime2 - startTime2;
  
              // Output the CUDA time just including adding operation time into file
              outputFile21 << vectorSize << "\t1024\t" << numBlocks << "\t" << elapsedTime2 << "\n";
  
              // Output the CUDA time including adding operation and memory time into file
              outputFile22 << vectorSize <<"\t1024\t" << numBlocks << "\t" << elapsedMilliseconds2.count() << "\n";
          }
  
  		// Output part result to see if the addition is right or not
          if (vectorSize == 16) {
              // Print vectors using the printVector function
              printVector(vectorA, "Vector A experiments 5, size == 16: :");
              printVector(vectorB, "Vector B experiments 5, size == 16: :");
              printVector(resultVector, "resultVector experiments 5, size == 16:");
  			      std::cout << std::endl;
          }
  
  	}
  	
  	
   	// Close files
    outputFile21.close();
    outputFile22.close();

    return 0;
}
