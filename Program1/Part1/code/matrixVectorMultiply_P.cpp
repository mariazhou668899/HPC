
/* Complile Steps: running using command line of windows system
 * 
 * Step1: install Openblas, pay attention to adding path in the system variables
 * Step2: open cmd, navigate to the file holding matrixVectorMultiply_P.cpp
 * Step2: input g++ matrixVectorMultiply_P.cpp -o thread.exe -L/path/to/include -L/path/to/lib -lopenblas
 * -----  For example: g++ matrixVectorMultiply_P.cpp -o thread.exe -I"D:/xzAssignment/software/OpenBLAS-0.3.26-x64/include" -L"D:/software/OpenBLAS-0.3.26-x64/lib" -lopenblas
 * Step3: input ¡°thread.exe¡±
 * 
*/

/*
 * Introduction: matrixVectorMultiply.cpp contains the implementations of Thread Solution.
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
 *  3) Part 4: Thread Solution
 *   ---- Get Y1 (= A * X) by dividing A * X into several parts, assign these parts to 4 threads.
 *   ---- Get FLOPs of Thread: threads' implementation time - elapsedMilliseconds1.
 *   ---- Calculate residual1 of Sequential solution and BLAS solution.
 *   ---- Check the resitual1 tolerance (= 1e-15), within tolerance is true, or not is false.
  *  4) Part 4: Partial Results Ouput
 *   ---- Print out Marix A, Vector X, Vector Z,flaps of BLAS, Vector Y1, consuming time of Thread Solution
 *   flaps of thread solution, residual, residual tolerance
 * 
 * 3. Others:
 *  1) outputFile:
 *     It will be generated under the current running program folder automatically.
 *     Every line of outputFile includes: FLAPs of BLAS, FLAPs of Threads, Resitual of Threads,
 *     Resitual of Threads Tolerance, which are separated by space.
 *  2) Number of Test Cases
 *     It will generate cases from starting by 10, up to 10000, increasement is 10.
 *  3) Output of screen:
 *     Because with the increase of dimensions, it is impossible to output all the results,
 *     the output of screen just shows dimension <= 60 result detail.
 *
 */

#include <iostream>
#include <cstdlib>
#include <thread>
#include <vector>
#include <chrono>
#include <cmath>
#include <cblas.h>
#include <fstream>
#include <mutex>
#include <windows.h>

using namespace std;


// Declear matrix A, vector X, vector Y1
vector<vector<double>> A;
vector<double> X;
vector<double> Y1; // For matrix-vector-multiply thread solution
vector<double> Z;



//==================================== Tool Part ======================================

//============================ Thread Solution Tool ============================= 
// Function to do Matrix-Vector multiplication for thread solution
void multyThread(int cols, int start, int end, int core) {

    // Set affinity for the current thread
    DWORD_PTR mask = 1ull << core;
    SetThreadAffinityMask(GetCurrentThread(), mask);

    for (int i = start; i < end; ++i) {
        Y1[i] = 0;
        for (int j = 0; j < cols; ++j) {
            Y1[i] += A[i][j] * X[j];

            //If want to see affinitation works, can cout this line
            //std::cout << "Thread " << core << ": Y1[" << i << "] = " << Y1[i] << std::endl;
        }
    }
}


//=================================== BLAS Tool ==================================== 
// Function to 2D tranfer matrix to 1D matrix
std::vector<double> flattenMatrix(const std::vector<std::vector<double>>& matrix) {
    std::vector<double> flattened;
    for (const auto& row : matrix) {
        flattened.insert(flattened.end(), row.begin(), row.end());
    }
    return flattened;
}


//=================================== Get Residual ==================================== 
// Function to calculate the residual between two vectors
double calculateResidual(const vector<double>& Y_BLAS, const vector<double>& Y_My) {
    // Assuming Y_BLAS and Y_My have the same size
    int n = Y_BLAS.size();

    // Calculate the residual
    double residual = 0.0;
    for (int i = 0; i < n; ++i) {
        double diff = Y_BLAS[i] - Y_My[i];
        residual += diff * diff;
    }

    return sqrt(residual);
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
    // File name will be "matrix_vector_thread_affini_4.txt" which will generate when running
    // in my 4-core computer and using Visual Studio complier; And it also will be 
    // "matrix_vector_thread_affini_8.txt" running on other 8-core computer and using Visual
    // Studio complier.
    ofstream outputFile("matrix_vector_thread_affini_g++_4.txt");

    // Check if the file is open
    if (!outputFile.is_open()) {
        cerr << "Unable to open output file.\n";
        return 0;
    }

    const double tolerance = 1e-6; // For residual tolerance checking
    bool withinTolerance = false; // For record Thread residual tolerance status

    for (int dimension = 10; dimension <= 10000; dimension += 10) {

        int rows = dimension;
        int cols = dimension;

        //===========================  Part 1 =========================== 
        //===================== Generate Test Case ======================
        
        // Initialize matrix A
        A.resize(rows, vector<double>(cols, 0.0));
        for (int i = 0; i < rows; ++i) {
            A[i].resize(cols);
        }

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                A[i][j] = 1.0000 + j % 3 ; // Generate random integer
            }
        }

        // Initialize vector X
        X.resize(cols);
        for (int j = 0; j < cols; ++j) {
            X[j] = 0.0001; // Generate random integer
        }


        // Initialize results vector Y1 
        Y1.resize(rows, 0.0);

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
        outputFile << flops << "         ";


        //===========================  Part 3 =========================== 
        //======================= Thread Solution =======================
        
        // Declaring four threads
        // Keep same thread number in my 4-core computer or other 8-core computer
        int num_threads = 8; 

        // Declaring threads
        vector<thread> threads;

        // Declaring core #
        int coreNumber = 0;

        // Measure the start time, Time (Milliseconds)
        auto startTime1 = chrono::high_resolution_clock::now();

        // Creating threads, each evaluating its own part
        int chunk_size = rows / num_threads;
        for (int i = 0; i < num_threads; ++i) {
            int start = i * chunk_size;
            int end = (i == num_threads - 1) ? rows : (i + 1) * chunk_size;
            threads.push_back(thread([=]() { multyThread(cols, start, end, coreNumber); }));
            // For 4-core computer, update coreNumber to ensure two threads share the same core
            coreNumber = (coreNumber + 1) % (num_threads / 2);

            // For 8-core computer, thread number is 8, 1 thread per core.
            //coreNumber;
        }

        // Joining and waiting for all threads to complete
        for (auto& t : threads) {
            t.join();
        }

        // Record metrix-vector-multiply thread solution running end time
        auto endTime1 = chrono::high_resolution_clock::now();

        // Calculate time consuming of metrix-vector-multiply thread solution
        chrono::duration<double, milli> elapsedMilliseconds1 = endTime1 - startTime1;

        // Write FLOPS of BLAS into the outputFile
        double flops1 = (2.0 * rows * cols) / (elapsedMilliseconds1.count() / 1000.0);

        // Write FLOPs of metrix-vector-multiply thread solution into the outputFile
        outputFile << flops1 << "  ";

        // Get residual of matrix-vector-multiply thread solution and BLAS
        double residual1 = 0.0;
        residual1 = calculateResidual(Z, Y1);

        // Write residual of  matrix-vector-multiply thread solution and BLAS into the outputFile
        outputFile << residual1 << "  ";


        // Check the tolerance
        if (residual1 < tolerance) {
            withinTolerance = true;
        }
        else {
            withinTolerance = false;
        }

        // Write residual tolerance status into into the outputFile, ture (1) for within tolerance
        //    false (0 ) for not
        outputFile << withinTolerance << "\n";

        //===========================  Part 4 =========================== 
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


            // Output the result calculated by matrix-vector-multiply thread solution
            cout << "    Thread Results: ";
            for (int i = 0; i < rows; ++i) {
                cout << " " << Y1[i];
            }
            cout << endl;
            cout << "    Time of Thread: " << elapsedMilliseconds1.count();
            cout << endl;
            cout << "    FLOPs of Thread: " << flops1 << endl;
            cout << "    Residual of Thread: " << residual1 << endl;

            // Check if the residual is within the tolerance
            if (residual1 < tolerance) {
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

    // Close the output file
    outputFile.close();

    return 0;
}
