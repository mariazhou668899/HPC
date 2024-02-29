/* Complile methods:
 * 1. running using Visual Studio IDE
 * Step1: install Openblas, pay attention to doing relevant configures in the Visual Studio.
 * Step2: Create a project, configure Openblas in this project.
 * Step3: Add matrixMatrixMultiply_B.cpp to the Resource Files of this project.
 * Step4: run program
 * Step5: check the results from screen or from the file which locates in the current project folder
 *
 * 2. running using command line and g++
 * Step1: install Openblas and gcc complier(include g++) in windows system, pay attention to adding path in the system variables
 * Step2: open cmd, navigate to the file holding matrixMatrixMultiply_B.cpp
 * Step3: input g++ matrixMatrixMultiply_B.cpp -o blas.exe -L/path/to/include -L/path/to/lib -lopenblas
 * -----  For example: g++ matrixMatrixMultiply_B.cpp -o blas.exe -I"D:/xzAssignment/software/OpenBLAS-0.3.26-x64/include" -L"D:/software/OpenBLAS-0.3.26-x64/lib" -lopenblas
 * Step4: input blas.exe
 * Step5: check the results from screen or from the file which is generated in the same folder of g++ matrixMatrixMultiply_B.cpp
 *
*/

/*  * Introduction: matrixMatrixMultiply_B.cpp contains the solution implementation.
 * 1. Tool functions: Provide tool functions for main() part
 *  1) generateA()
 *   ---- Function to generate and return a vector of random double values with 0 decimals
 *  2) generateB()
 *   ---- Function to generate and return a vector of value 0.0001
 *  3) printMatrix()
 *   ---- Function to print a matrix
 *
 * 2. Others:
 *  1) outputFile:
 *     It will be generated in the current running program folder automatically if running by Visual Studio, or
 *     in the same folder of matrixMatrixMultiply_B.cpp by g++.
 *     Every line of outputFile includes: dimension, FLAPs of BLAS.
 *  2) Number of Test Cases
 *     It will generate cases from starting by 10, up to 10000: 10 – 1000: increasement is 1;1000 – 5000: increasement is 50;
 *     5000 – 8000: increasement is 200;8000 – 10000: increasement is 400.
 *
 */


#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cblas.h>
#include <iomanip>

using namespace std;


// Function to generate and return a vector of random double values between 0 and 1
std::vector<double> generateA(int size) {
    std::vector<double> vec(size);
    for (int i = 0; i < size; ++i) {
        vec[i] = 1.0000 + i % 3;
    }
    return vec;
}

// Function to generate and return a vector of random double values between 0 and 1
std::vector<double> generateB(int size) {
    std::vector<double> vec(size);
    for (int i = 0; i < size; ++i) {
        vec[i] = 0.0001;
    }
    return vec;
}

// Function to print a matrix
void printMatrix(const std::vector<double>& matrix, int numCols) {
    size_t numRows = matrix.size() / numCols;

    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            std::cout << matrix[i * numCols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
	
    // Open the file for writing the results	
	// "mm_b_g++_4C_1000.txt" for 10 - 1000 with step 1
	// "mm_b_g++_4C_5000.txt" for 1000 - 5000 with step 50
	// "mm_b_g++_4C_8000.txt" for 5000 - 8000 with step 200
	// "mm_b_g++_4C_10000.txt" for 8000 - 10000 with step 400
    ofstream outputFile("mm_b_g++_4C_1000.txt");

    if (!outputFile.is_open()) {
        cerr << "Unable to open output file.\n";
        return 0;
    }

    int rows = 0;
	// Change according to data file
    for (int dimension = 10; dimension <= 1000; dimension += 1) {
		// Declear and initialize matrix A, B, C_Blas, C_My
        vector<double> A = generateA(dimension * dimension);
        vector<double> B = generateB(dimension * dimension);
        vector<double> C_Blas(dimension * dimension, 0.0);
        vector<double> C_My(dimension * dimension, 0.0);
		
		// Write dimension into outputFile
        outputFile << dimension << "    ";
		
        rows = dimension;
		
		// Calculate C_Blas = A * B, and calculate Blas' flops
        auto startTime = chrono::high_resolution_clock::now();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dimension, dimension, dimension, 1.0, A.data(), dimension, B.data(), dimension, 0.0, C_Blas.data(), dimension);
        auto endTime = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> elapsedMilliseconds = endTime - startTime;
        double flops = (2.0 * dimension * dimension * dimension) / (elapsedMilliseconds.count() / 1000.0);
		// Write flops of blas into outputFile
        outputFile << flops << "\n";

        if (rows <= 15) {
            cout << "CASE DIMENSION: " << dimension << ":\n" << endl;

            // Print out matrix
            cout << "Marix A:" << "\n" << endl;
            printMatrix(A, rows);
            cout << "\n";

            // Print out vector
            cout << "Marix B:" << " " << endl;
            printMatrix(B, rows);
            cout << "\n";

            // Output the result calculated by cblas_dgemv of BLAS
            cout << "    BLAS Results: \n";
            printMatrix(C_Blas, rows);

            cout << "    Time of BLAS: " << elapsedMilliseconds.count();
            cout << endl;
            cout << "    FLOPs of BLAS: " << flops << endl << endl;

        }
        else if (rows == 300 || rows == 500 || rows == 1000){
            cout << "CASE DIMENSION: " << dimension << ":\n" << endl;

            // Output the result calculated by cblas_dgemv of BLAS
            cout << "    BLAS Results: \n";
            cout << "    Time of BLAS: " << elapsedMilliseconds.count();
            cout << endl;
            cout << "    FLOPs of BLAS: " << flops << endl << endl;
            cout << "\n";
        }
    }
    outputFile.close();
    return 0;
}
