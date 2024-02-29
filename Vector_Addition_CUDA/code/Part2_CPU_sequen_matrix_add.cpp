/* 1. Complile methods: running using UW GPU computer
 * Step1: navigate to the file holding current file
 * Step3: input g++ -o cpu_sequen Part2_CPU_sequen_matrix_add.cpp
 * Step4: input ./cpu_seqen
 *
*/

/* 2. Introduction: This program is to realize vector addition using CPU sequential way.
 * It tests 2^3 to 2^20 size vector addition, and store the operation time into file.
 *
*/


#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <chrono>


// Function to generate test cases
void generateRandomVector(std::vector<int>& vec, size_t size) {
    vec.resize(size);
    for (size_t i = 0; i < size; ++i) {
        // Generate random integer between -10 and 10
        vec[i] = rand() % 21 - 10;
    }
}

// Functoin to do the sequential vector-addition
void vectorAddition(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& result) {
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
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

int main() {
    // Seed for reproducibility
    srand(static_cast<unsigned>(time(nullptr)));

    // Open a file to write results
    std::ofstream outputFile("vector_sequen_addition_results.txt");

    // Perform experiments for vector sizes from 2^3 to 2^20
    for (size_t i = 3; i <= 20; i += 1) {
        size_t vectorSize = 1 << i;

        // Generate vectors with random integers between -10 and 10
        std::vector<int> vectorA, vectorB, resultVector;
        generateRandomVector(vectorA, vectorSize);
        generateRandomVector(vectorB, vectorSize);
        resultVector.resize(vectorSize, 0);

        if (vectorSize == 8) {
            // Print vectorA before kernel execution
            printVector(vectorA, "Vector A Before experiments: ");

            // Print vectorB before kernel execution
            printVector(vectorB, "Vector B Before experiments: ");

            // Print resultVector before kernel execution
            printVector(resultVector, "resultVector Before experiments: ");
        }

		// Record the start time including memory and operation 
		auto startTime3 = std::chrono::high_resolution_clock::now();

        // Perform vector addition
        vectorAddition(vectorA, vectorB, resultVector);
            
		// Record the end time just including memory
		auto endTime3 = std::chrono::high_resolution_clock::now();
		
		// Calculate CUDA vector addtion time including memory and adding operation
		std::chrono::duration<double, std::micro> elapsedMilliseconds3 = endTime3 - startTime3;

        
        // Write vectorSize, elapsedSeconds to the file
        outputFile << vectorSize << "\t" << elapsedMilliseconds3.count() << std::endl;

        // Print partial results
        if (vectorSize == 8) {
          // Print vectors using the printVector function
            printVector(vectorA, "Vector A after all the experiments:");
            printVector(vectorB, "Vector B after all the experiments:");
            printVector(resultVector, "resultVector after all the experiments:");
        }



    }

    // Close the file
    outputFile.close();

    return 0;
}
