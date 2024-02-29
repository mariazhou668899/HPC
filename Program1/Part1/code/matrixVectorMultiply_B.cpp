/* Complile Steps: running using command line of windows system
 *
 * Step1: install Openblas, pay attention to adding path in the system variables
 * Step2: open cmd, navigate to the file holding matrixVectorMultiply_B.cpp
 * Step2: input g++ matrixVectorMultiply_B.cpp -o blas.exe -L/path/to/include -L/path/to/lib -lopenblas
 * -----  For example: g++ matrixVectorMultiply_B.cpp -o blas.exe -I"D:/xzAssignment/software/OpenBLAS-0.3.26-x64/include" -L"D:/software/OpenBLAS-0.3.26-x64/lib" -lopenblas
 * Step3: input ¡°blas.exe¡±
 *
*/
/*  * Introduction: matrixVectorMultiply.cpp contains the implementations of Thread Solution.
 * 1. Tool part: Provide tool functions for main() part
 *  1) Thread Solution Tool
 *   ---- multyThread() is to do Matrix-Vector multiplication for thread solution.
 *  2) BLAS Tool
 *   ---- flattenMatrix() to tranfer 2D matrix to 1D matrix.
 *  3) Print Out Tool
 *   ---- printMatrix() to print out 2D matrix into screen.
 *   ---- printVector() to print out vector into screen.
 *
 * 2. Main part: implementation program to get the analysis data
 *  1) Part 1: Generate test case, including filling 2D-matrix A, vector X.
 *   ---- Use rand() % 10 to fill integer into 2D-matrix A.
 *   ---- use static_cast<double>(rand()) / RAND_MAX to fill random floating-point number between 0 and 1.
 *  2) Part 2: BLAS Solution.
 *   ---- Get Z after A * X by calling cblas_dgemv(),
 *        Z will be the right result reference for other solutions,
 *        and will be used to calculate residul and residual tolerance
 *   ---- Get FLOPs of BLAS: Calculate cblas_dgemv() implementation time - elapsedMilliseconds,
 *        then, calculate FLOPs of BLAS = (2.0 * rows * cols) / (elapsedMilliseconds.count() / 1000.0).
  *  4) Part 3: Partial Results Ouput
 *   ---- Print out Marix A, Vector X, Vector Z,flaps of BLAS, Vector Y1, consuming time of Thread Solution
 *   flaps of thread solution, residual, residual tolerance
 *
 * 3. Others:
 *  1) outputFile:
 *     It will be generated under the current running program folder automatically.
 *     Every line of outputFile includes: FLAPs of BLAS.
 *  2) Number of Test Cases
 *     It will generate cases from starting by 10, up to 10000, increasement is 10.
 *  3) Output of screen:
 *     Because with the increase of dimensions, it is impossible to output all the results,
 *     the output of screen just shows dimension <= 60 result detail.
 *
 */

#include <iostream>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cmath>
#include <cblas.h>
#include <fstream>

using namespace std;


//=================================== BLAS Tool ==================================== 
// Function to 2D tranfer matrix to 1D matrix
std::vector<double> flattenMatrix(const std::vector<std::vector<double>>& matrix) {
    std::vector<double> flattened;
    for (const auto& row : matrix) {
        flattened.insert(flattened.end(), row.begin(), row.end());
    }
    return flattened;
}


//=================================== Print Out Tools ==================================== 
// Function to print a matrix
void printMatrix(const vector<vector<double>>& matrix) {
    for (const auto& row : matrix) {
        for (double value : row) {
            cout << value << " ";
        }
        cout << endl;
    }
}

// Function to print a vector
void printVector(const vector<double>& vec) {
    for (double value : vec) {
        cout << value << " ";
    }
    cout << endl;
}

//==================================== Main Part ======================================

int main() {

    // Open the file for writing the results
    // File name will be "matrix_vector_blas_4.txt" when running
    // in my 4-core computer and using Visual Studio complier; And it also will be 
    // "matrix_vector_blas_4.txt" when running on other 8-core computer and 
    // using Visual Studio complier.
    ofstream outputFile("matrix_vector_blas_g++_4.txt");

    // Check if the file is open
    if (!outputFile.is_open()) {
        cerr << "Unable to open output file.\n";
        return 0;
    }

    for (int dimension = 10; dimension <= 10000; dimension += 10) {

        int rows = dimension;
        int cols = dimension;

        //===========================  Part 1 =========================== 
        //===================== Generate Test Case ======================

        // Declear matrix A, vector X, vector Y1
        vector<vector<double>> A;
        vector<double> X;
        vector<double> Z;

        // Initialize matrix A
        A.resize(rows, vector<double>(cols, 0.0));
        for (int i = 0; i < rows; ++i) {
            A[i].resize(cols);
        }

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                A[i][j] = 1.0000 + j % 3; // Generate random integer
            }
        }

        // Initialize vector X
        X.resize(cols);
        for (int j = 0; j < cols; ++j) {
            X[j] = 0.0001; // Generate random integer
        }

        // Initialize results vector Y  
        Z.resize(rows, 0.0);

        //===========================  Part 2 =========================== 
        //======================== BLAS Solution =======================

        // Calculate FLOPs, residual of matrix-vector-multiply thread solution and BLAS

        // Flatten matrix A for cblas_dgemv
        std::vector<double> flattenedA = flattenMatrix(A);

        //// Measure the start time, Time (Milliseconds)
        auto startTime = chrono::high_resolution_clock::now();

        //// Using CBLAS for vector-matrix multiplication: Z = alpha * A * X + beta * Z
        cblas_dgemv(CblasRowMajor, CblasNoTrans, rows, cols, 1, flattenedA.data(), cols, X.data(), 1, 0, Z.data(), 1);


        //// Measure the end time
        auto endTime = chrono::high_resolution_clock::now();

        //// Calculate time consuming of BLAS
        chrono::duration<double, milli> elapsedMilliseconds = endTime - startTime;

        //// Calculate FLOPs of BLAS
        double flops = (2.0 * rows * cols) / (elapsedMilliseconds.count() / 1000.0);

        // Write FLOPS to the output file
        outputFile << flops << "\n";

        //===========================  Part 3 =========================== 
        //===================== Partial Results Ouput ===================

        // Output test case details to screen when dimension <= 60
        if (rows <= 60) {

            cout << "CASE DIMENSION: " << dimension << ":\n" << endl;

            // Print out matrix
            cout << "Marix A:" << "\n" << endl;
            printMatrix(A);
            cout << "\n";

            // Print out vector
            cout << "Vector X:" << " " << endl;
            printVector(X);
            cout << "\n";

            // Output the result calculated by cblas_dgemv of BLAS
            cout << "    BLAS Results: ";
            for (int i = 0; i < rows; ++i) {
                cout << Z[i] << " ";
            }
            cout << "\n";
            cout << "    Time of BLAS: " << elapsedMilliseconds.count();
            cout << endl;
            cout << "    FLOPs of BLAS: " << flops << endl << endl;

            cout << "\n";
        }

    }

    // Close the output file
    outputFile.close();

    return 0;
}
