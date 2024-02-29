/* Complile methods:
 * 1. running using Visual Studio IDE
 * Step1: install Openblas, pay attention to doing relevant configuring in the Visual Studio.
 * Step2: Create a project, configure Openblas in this project.
 * Step3: Add mmM_S_staticBlock.cpp to the Resource Files of this project.
 * Step4: run program
 * Step5: check the results from screen or from the file which locates in the current project folder
 *
 * 2. running using command line and g++
 * Step1: install Openblas, pay attention to adding path in the system variables
 * Step2: open cmd, navigate to the file holding mmM_S_staticBlock.cpp
 * Step3: input g++ mmM_S_staticBlock.cpp -o s_staticBlock.exe -L/path/to/include -L/path/to/lib -lopenblas
 * -----  For example: g++ mmM_S_staticBlock.cpp -o s_staticBlock.exe -I"D:/xzAssignment/software/OpenBLAS-0.3.26-x64/include" -L"D:/software/OpenBLAS-0.3.26-x64/lib" -lopenblas
 * Step4: input s_staticBlock.exe
 * Step5: check the results from screen or from the file which is generated in the same folder of g++ mmM_S_staticBlock.cpp
 *
*/

/*  * Introduction: mmM_S_staticBlock.cpp contains the solution implementation.
 * 1. Tool functions: Provide tool functions for main() part
 *  1) calculateResidual()
 *   ---- Function to calculate the residual between two vectors.
 *  2) multySequentialBlocked()
 *   ---- Function to do block-wise Matrix-Matrix multiplication.
 *  3) generateA()
 *   ---- Function to generate and return a vector of random double values with 0 decimals
 *  4) generateB()
 *   ---- Function to generate and return a vector of value 0.0001
 *  5) printMatrix()
 *   ---- Function to print a matrix
 *
 * 2. Others:
 *  1) outputFile:
 *     It will be generated in the current running program folder automatically if running by Visual Studio, or
 *     in the same folder of mmM_S_staticBlock.cpp by g++.
 *     Every line of outputFile includes: FLAPs of BLAS, FLAPs of my solution, Resitual of my solution,
 *     Resitual of my solution Tolerance.
 *  2) Number of Test Cases
 *     It will generate cases from starting by 10, up to 1000, increasement is 1.
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

constexpr int STATIC_BLOCK_SIZE = 64;  // blockSize < 512; sqrt(L1_CACHE_SIZE / 3.0) = 512, my computer's L1_CACHE_SIZE is 256K

// Function to calculate the residual between two vectors
double calculateResidual(const vector<double>& Y_BLAS, const vector<double>& Y_My) {
    double residual = 0.0;
    for (size_t i = 0; i < Y_BLAS.size(); ++i) {
        double diff = fabs(Y_BLAS[i] - Y_My[i]);
        residual += diff;
    }
    return residual;
}

// Function to do block-wise Matrix-Matrix multiplication
void multySequentialBlocked(const vector<double>& A, const vector<double>& B, vector<double>& C, int dimension, int blockSize) {
    for (int ii = 0; ii < dimension; ii += blockSize) {
        for (int jj = 0; jj < dimension; jj += blockSize) {
            for (int kk = 0; kk < dimension; kk += blockSize) {
                for (int i = ii; i < min(ii + blockSize, dimension); ++i) {
                    for (int j = jj; j < min(jj + blockSize, dimension); ++j) {
                        for (int k = kk; k < min(kk + blockSize, dimension); ++k) {
                            C[i * dimension + j] += A[i * dimension + k] * B[k * dimension + j];
                        }
                    }
                }
            }
        }
    }
}


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

    //Open the file for writing the results
    ofstream outputFile("mmm_s_staticBlock_g++_4C_1000.txt");

    if (!outputFile.is_open()) {
        cerr << "Unable to open output file.\n";
        return 0;
    }

    int rows = 0;
    const double tolerance = 1e-6;
    bool withinTolerance = false;
    int blockSize = STATIC_BLOCK_SIZE;

    for (int dimension = 10; dimension <= 1000; dimension += 1) {

        // Declear and initialize matrix A, B, C_Blas, C_My
        vector<double> A = generateA(dimension * dimension);
        vector<double> B = generateB(dimension * dimension);
        vector<double> C_Blas(dimension * dimension, 0.0);
        vector<double> C_My(dimension * dimension, 0.0);

        rows = dimension;

        // Write dimension into outputFile
        outputFile << dimension << "      ";

        // Calculate C_Blas = A * B, and calculate Blas' flops
        auto startTime = chrono::high_resolution_clock::now();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dimension, dimension, dimension, 1.0, A.data(), dimension, B.data(), dimension, 0.0, C_Blas.data(), dimension);
        auto endTime = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> elapsedMilliseconds = endTime - startTime;
        double flops = (2.0 * dimension * dimension * dimension) / (elapsedMilliseconds.count() / 1000.0);
        
        // Write blas flops into outputFile
        outputFile << flops << "         ";

        // Calculate My_Blas = A * B, and calculate Standard Sequential solution' flops
        startTime = chrono::high_resolution_clock::now();
        multySequentialBlocked(A, B, C_My, dimension, blockSize);
        endTime = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> elapsedMilliseconds1 = endTime - startTime;
        double flops1 = (2.0 * dimension * dimension * dimension) / (elapsedMilliseconds1.count() / 1000.0);

        // Write my solution flops into outputFile
        outputFile << flops1 << "  ";

        // Calculate residual between Standard Sequential solution and benchmark BLAS
        double residual = calculateResidual(C_Blas, C_My);

        // Write my solution residual into outputFile
        outputFile << residual << "  ";

        // Get residual status and write into outputFile
        if (residual < tolerance) {
            withinTolerance = true;
        }
        else {
            withinTolerance = false;
        }
        outputFile << withinTolerance << "\n";


        // Output results to the screen
        if (rows <= 15) {
            cout << "CASE DIMENSION: " << dimension << ":\n" << endl;

            // Print out matrix A
            cout << "Marix A:" << "\n" << endl;
            printMatrix(A, rows);
            cout << "\n";

            // Print out matrix B
            cout << "Marix B:" << " " << endl;
            printMatrix(B, rows);
            cout << "\n";

            // Output the result calculated by cblas_dgemm of BLAS
            cout << "    BLAS Results: \n";
            printMatrix(C_Blas, rows);

            cout << "    Time of BLAS: " << elapsedMilliseconds.count();
            cout << endl;
            cout << "    FLOPs of BLAS: " << flops << endl << endl;

            //Output the result calculated by matrix-matrix-multiply sequential solution 
            cout << "    Block Sequential Results: \n";
            printMatrix(C_My, rows);

            cout << endl;
            cout << "    Time of Block Sequential: " << elapsedMilliseconds1.count();
            cout << endl;
            cout << "    FLOPs of Block Sequential: " << flops1 << endl;
            cout << "    Residual of Block Sequential: " << residual << endl;

            // Check if the residual is within the tolerance
            if (residual < tolerance) {
                // Results match within the tolerance
                cout << "    Is results match within tolerance? " << "YES" << endl << endl;
            }
            else {
                // Results do not match within the tolerance
                cout << "    Is results match within tolerance? " << "NO" << endl << endl;
            }
            cout << "\n";
        }
        else if (rows == 300 || rows == 500 || rows == 1000) {
            cout << "CASE DIMENSION: " << dimension << ":\n" << endl;

            // Output the result calculated by cblas_dgemm of BLAS
            cout << "    BLAS Results: \n";

            cout << "    Time of BLAS: " << elapsedMilliseconds.count();
            cout << endl;
            cout << "    FLOPs of BLAS: " << flops << endl << endl;

            //Output the result calculated by matrix-matrix-multiply sequential solution 
            cout << "    Static Block Sequential Results: \n";

            cout << endl;
            cout << "    Time of Static Block Sequential: " << elapsedMilliseconds1.count();
            cout << endl;
            cout << "    FLOPs of Static Block Sequential: " << flops1 << endl;
            cout << "    Residual of Static Block Sequential: " << residual << endl;

            // Check if the residual is within the tolerance
            if (residual < tolerance) {
                // Results match within the tolerance
                cout << "    Is results match within tolerance? " << "YES" << endl << endl;
            }
            else {
                // Results do not match within the tolerance
                cout << "    Is results match within tolerance? " << "NO" << endl << endl;
            }
            cout << "\n";
        }
    }

    outputFile.close();
    return 0;
}
