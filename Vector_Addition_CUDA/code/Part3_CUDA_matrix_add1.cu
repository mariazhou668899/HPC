/* 1. Complile methods: running using UW GPU computer
 * Step1: Login the GPU computer
 * Step1: navigate to the file holding current file
 * Step3: input nvcc -o addition1 Part3_CUDA_matrix_add1.cu
 * Step4: input ./addition1
 *
*/

/* 2. In this program, it will conduct 5 experiment: 
  1) Experiment 1: configurations. Varying #Threads, #block 1, increase Threads
    std::vector<size_t> numThreadsList1 = {8, 64, 128, 512};
	  std::vector<size_t> size1 = {3,6,7,9}, 2^;
    std::ofstream outputFile11("experiment1_thread_op_11.txt");
    std::ofstream outputFile12("experiment1_thread_m_12.txt");

  2) Experiment 2: configurations. Varying #block, #Threads 1024, increase block
    std::vector<size_t> numBlocksList2 = {1, 8,64,128};
    std::vector<size_t> size2 = {10,13,16,17}, 2^;
    std::ofstream outputFile21("experiment2_block_op_21.txt");
    std::ofstream outputFile22("experiment2_block_m_22.txt");

  3) Experiment 3: configurations. Varying #block and #Threads with the same number
    std::vector<size_t> numThreadsAndBlock3 = {1, 2, 4, 64};
    std::vector<size_t> size3 = {5,8,10,15}, 2^;
    std::ofstream outputFile31("experiment3_thread_block_op_31.txt");
    std::ofstream outputFile32("experiment3_thread_block_m_32.txt");
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

    // Experiment 1: configurations. Varying #Threads, #block 1, increase Threads
    std::vector<size_t> numThreadsList1 = {8, 64, 128, 512};
	  std::vector<size_t> size1 = {3,6,7,9};
    std::ofstream outputFile11("experiment1_thread_op_11.txt");
    std::ofstream outputFile12("experiment1_thread_m_12.txt");

    // Experiment 2: configurations. Varying #block, #Threads 1024, increase block
    std::vector<size_t> numBlocksList2 = {1, 8,64,128};
	  std::vector<size_t> size2 = {10,13,16,17};
    std::ofstream outputFile21("experiment2_block_op_21.txt");
    std::ofstream outputFile22("experiment2_block_m_22.txt");

    // Experiment 3: configurations. Varying #block and #Threads with the same number
    std::vector<size_t> numThreadsAndBlock3 = {1, 2, 4, 64};
	  std::vector<size_t> size3 = {5,8,10,15};
    std::ofstream outputFile31("experiment3_thread_block_op_31.txt");
    std::ofstream outputFile32("experiment3_thread_block_m_32.txt");
    
    
  	// Experiment 1: Varying #Threads, #block 1
  	// 	size1 = {3,6,7,9};
  	for (int i = 0; i < size1.size(); i++) {
  		
		// Calculate vectorsize
		size_t exponent = size1[i];
		size_t vectorSize = std::pow(2, exponent);

		  
		// Creat and initialize vectorA, vectorB and resultVector
		std::vector<int> vectorA, vectorB;
		generateRandomVector(vectorA, vectorSize);
		generateRandomVector(vectorB, vectorSize);
		std::vector<int> resultVector(vectorSize, 0);
  
		if (vectorSize == 8) {
			// Print vectorA before kernel execution
			printVector(vectorA, "Vector A Before experiment 1, size = 8:  ");

			// Print vectorB before kernel execution
			printVector(vectorB, "Vector B Before experiment 1, size = 8:  ");

			// Print resultVector before kernel execution
			printVector(resultVector, "resultVector Before experiment 1, size = 8: ");
			std::cout << std::endl;
		}
  		
		// Experiment 1: Varying #Threads, #block 1
		for (size_t j = 0; j < numThreadsList1.size(); j++) {
		  
			// Get the threads number per block
			size_t numThreads = numThreadsList1[j];

			// Record the start time including memory and operation 
			auto startTime1 = std::chrono::high_resolution_clock::now(); 

			// Calculate CUDA vector addition and return the adding operation time
			float elapsedTime1 = vectorAdditionCUDA(vectorA, vectorB, resultVector, 1, numThreads);

			// Record the end time just including memory
			auto endTime1 = std::chrono::high_resolution_clock::now();

			// Calculate CUDA vector addtion time including memory and adding operation
			std::chrono::duration<double, std::milli> elapsedMilliseconds1 = endTime1 - startTime1;

			// Output the CUDA time just including adding operation time into file
			outputFile11 << vectorSize << "\t" << numThreads << "\t1\t" << elapsedTime1 << "\n";

			// Output the CUDA time including adding operation and memory time into file
			outputFile12 << vectorSize << "\t" << numThreads << "\t1\t" << elapsedMilliseconds1.count() << "\n";
		}
  
		// Output part result to see if the addition is right or not
		if (vectorSize == 8) {
			// Print vectors using the printVector function
			printVector(vectorA, "Vector A after experiment, size = 8: ");
			printVector(vectorB, "Vector B after experiment, size = 8: ");
			printVector(resultVector, "resultVector after experiment, size = 8: ");
			std::cout << std::endl;
		}
  
  	} 	
  
    
   	// Experiment 2: Varying #Blocks, #Threads 1024
  	// size2 = {10,13,16,17};
  	for (int i = 0; i < size2.size(); i++) {
  		
		// Calculate vectorsize
  		size_t exponent = size2[i];
  		size_t vectorSize = std::pow(2, exponent);
          
		// Creat and initialize vectorA, vectorB and resultVector
		std::vector<int> vectorA, vectorB;
		generateRandomVector(vectorA, vectorSize);
		generateRandomVector(vectorB, vectorSize);
		std::vector<int> resultVector(vectorSize, 0);

		if (vectorSize == 1024) {
			// Print vectorA before kernel execution
			printVector(vectorA, "Vector A Before experiment 2, size = 1024:  ");

			// Print vectorB before kernel execution
			printVector(vectorB, "Vector B Before experiment 2, size = 1024:  ");

			// Print resultVector before kernel execution
			printVector(resultVector, "resultVector Before experiment 2, size = 1024:  ");
			std::cout << std::endl;
		}
          
		// numBlocksList2 = {1, 8,64,128};  
		for (size_t j = 0; j < numBlocksList2.size(); j++) {
		  
			// Get the block number
			size_t numBlocks = numBlocksList2[j];
			  
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
		if (vectorSize == 1024) {
			// Print vectors using the printVector function
			printVector(vectorA, "Vector A after experiment 2, size = 1024:");
			printVector(vectorB, "Vector B after experiment 2, size = 1024:");
			printVector(resultVector, "resultVector after experiment 2, size = 1024:");
			std::cout << std::endl;
		}
  
	}  
      	

  	 
  	// Experiment 3: Varying #Blocks, #Threads 1024
  	// size3 = {5,8,10,15};
  	for (int i = 0; i < size3.size(); i++) {
  		
		// Calculate vectorsize
		size_t exponent = size3[i];
		size_t vectorSize = std::pow(2, exponent);
		  
		// Creat and initialize vectorA, vectorB and resultVector
		std::vector<int> vectorA, vectorB;
		generateRandomVector(vectorA, vectorSize);
		generateRandomVector(vectorB, vectorSize);
		std::vector<int> resultVector(vectorSize, 0);
  
        if (vectorSize == 32) {
			// Print vectorA before kernel execution
			printVector(vectorA, "Vector A Before experiment 3, size = 32: ");

			// Print vectorB before kernel execution
			printVector(vectorB, "Vector B Before experiment 3, size = 32:");

			// Print resultVector before kernel execution
			printVector(resultVector, "resultVector Before experiment 3, size = 32:");
			std::cout << std::endl;
        }
  
  		// Experiment 3: Varying #Threads and #Blocks
        for (size_t j = 0; j < numThreadsAndBlock3.size(); ++j) {
            
            // Get the thread number and block number
            size_t numThreads3 = numThreadsAndBlock3[j];
            size_t numBlocks3 = numThreadsAndBlock3[j];
            
            // Record the start time including memory and operation 
            auto startTime3 = std::chrono::high_resolution_clock::now();
            
            // Calculate CUDA vector addition and return the adding operation time
            float elapsedTime3 = vectorAdditionCUDA(vectorA, vectorB, resultVector, numBlocks3, numThreads3);
            
            // Record the end time just including memory
            auto endTime3 = std::chrono::high_resolution_clock::now();
            
            // Calculate CUDA vector addtion time including memory and adding operation
            std::chrono::duration<double, std::milli> elapsedMilliseconds3 = endTime3 - startTime3;
    
            // Output the CUDA time just including adding operation time into file
            outputFile31 << vectorSize << "\t" << numThreads3 << "\t" << numBlocks3 << "\t" << elapsedTime3 << "\n";
    
            // Output the CUDA time including adding operation and memory time into file
            outputFile32 << vectorSize << "\t" <<numThreads3  << "\t" << numBlocks3 << "\t" << elapsedMilliseconds3.count() << "\n";
        }
  
  		  // Output part result to see if the addition is right or not
        if (vectorSize == 32) {
			// Print vectors using the printVector function
			printVector(vectorA, "Vector A after experiment 3, size = 32:");
			printVector(vectorB, "Vector B after experiment 3, size = 32:");
			printVector(resultVector, "resultVector after experiment 3, size = 32:");
			std::cout << std::endl;			
        }
  
  	 }
   
  	
  	
   	// Close files
    outputFile11.close();
    outputFile12.close();
    outputFile21.close();
    outputFile22.close();
    outputFile31.close();
    outputFile32.close();	

    return 0;
}
