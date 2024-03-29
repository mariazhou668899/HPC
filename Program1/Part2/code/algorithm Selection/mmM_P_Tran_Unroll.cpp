/* Complile methods:
 * 1. running using Visual Studio IDE
 * Step1: install Openblas, pay attention to doing relevant configuring in the Visual Studio.
 * Step2: Create a project, configure Openblas in this project.
 * Step3: Add mmM_P_Tran_Unroll.cpp to the Resource Files of this project.
 * Step4: run program
 * Step5: check the results from screen or from the file which locates in the current project folder
 *
 * 2. running using command line and g++
 * Step1: install Openblas, pay attention to adding path in the system variables
 * Step2: open cmd, navigate to the file holding mmM_P_Tran_Unroll.cpp
 * Step3: input g++ mmM_P_Tran_Unroll.cpp -o s_Tran_staticBlock.exe -L/path/to/include -L/path/to/lib -lopenblas
 * -----  For example: g++ mmM_P_Tran_Unroll.cpp -o s_Tran_staticBlock.exe -I"D:/xzAssignment/software/OpenBLAS-0.3.26-x64/include" -L"D:/software/OpenBLAS-0.3.26-x64/lib" -lopenblas
 * Step4: input s_Tran_staticBlock.exe
 * Step5: check the results from screen or from the file which is generated in the same folder of g++ mmM_P_Tran_Unroll.cpp
 *
*/

/*  * Introduction: mmM_P_Tran_Unroll.cpp contains the solution implementation.
 * 1. Tool functions: Provide tool functions for main() part
 *  1) calculateResidual()
 *   ---- Function to calculate the residual between two vectors.
 *  2) transpose()
 *   ---- Function to transpose matrix
 *  3) multySequentialUnrolledSingleThread
 *   ---- Function to do Matrix-Matrix multiplication with loop unrolling (single thread)
 *  4) multySequentialUnrolledMultiThread
 *   ---- Function to do Matrix-Matrix multiplication with loop unrolling (multi-threaded)
 *  5) generateA()
 *   ---- Function to generate and return a vector of random double values with 0 decimals
 *  6) generateB()
 *   ---- Function to generate and return a vector of value 0.0001
 *  7) printMatrix()
 *   ---- Function to print a matrix
 *
 * 2. Others:
 *  1) outputFile:
 *     It will be generated in the current running program folder automatically if running by Visual Studio, or
 *     in the same folder of mmM_P_Tran_Unroll.cpp by g++.
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
#include <thread>
#include <mutex>

using namespace std;

// Global mutex for synchronization
std::mutex mtx;

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

// Function to transpose a matrix
void transpose(std::vector<double>& matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = i + 1; j < size; ++j) {
            // Swap elements across the main diagonal
            std::swap(matrix[i * size + j], matrix[j * size + i]);
        }
    }
}

// Function to do Matrix-Matrix multiplication with loop unrolling (single thread)
void multySequentialUnrolledSingleThread(const vector<double>& A, const vector<double>& B, vector<double>& C, 
    int dimension, int startRow, int endRow) {
    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < dimension; ++j) {
            int idx = i * dimension + j;
            double sum = 0.0;

            // Loop unrolling with step size 4
            for (int k = 0; k < dimension; k += 4) {
                if (k + 3 < dimension) {
                    sum += A[i * dimension + k] * B[k * dimension + j];
                    sum += A[i * dimension + k + 1] * B[(k + 1) * dimension + j];
                    sum += A[i * dimension + k + 2] * B[(k + 2) * dimension + j];
                    sum += A[i * dimension + k + 3] * B[(k + 3) * dimension + j];
                }
                else {
                    // Handle the case when k + 3 exceeds dimension
                    int remainingElements = dimension - k;
                    for (int l = 0; l < remainingElements; ++l) {
                        sum += A[i * dimension + k + l] * B[(k + l) * dimension + j];
                    }
                }
            }

            mtx.lock();  // Lock before modifying the shared resource (C)
            C[idx] += sum;
            mtx.unlock();  // Unlock after modifying the shared resource
        }
    }
}

// Function to do Matrix-Matrix multiplication with loop unrolling (multi-threaded)
void multySequentialUnrolledMultiThread(const vector<double>& A, const vector<double>& B, vector<double>& C, 
    int dimension, int numThreads) {
    vector<thread> threads;

    int rowsPerThread = dimension / numThreads;

    for (int i = 0; i < numThreads; ++i) {
        int startRow = i * rowsPerThread;
        int endRow = (i == numThreads - 1) ? dimension : (i + 1) * rowsPerThread;

        threads.emplace_back(multySequentialUnrolledSingleThread, std::ref(A), std::ref(B), std::ref(C), 
            dimension, startRow, endRow);
    }

    // Join the threads to wait for them to finish
    for (auto& t : threads) {
        t.join();
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
	// "mmm_p_trans_unroll_g++_4C_1000.txt" for 10 - 1000 with step 1
    ofstream outputFile("mmm_p_trans_unroll_g++_4C_1000.txt");

    if (!outputFile.is_open()) {
        cerr << "Unable to open output file.\n";
        return 0;
    }

    int rows = 0;
    for (int dimension = 10; dimension <= 1000; dimension += 1) {
        vector<double> A = generateA(dimension * dimension);
        vector<double> B = generateB(dimension * dimension);
        vector<double> C_Blas(dimension * dimension, 0.0);
        vector<double> C_My(dimension * dimension, 0.0);
        rows = dimension;
        outputFile << dimension << "      ";

        auto startTime = chrono::high_resolution_clock::now();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dimension, dimension, dimension, 1.0, A.data(), 
            dimension, B.data(), dimension, 0.0, C_Blas.data(), dimension);
        auto endTime = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> elapsedMilliseconds = endTime - startTime;
        double flops = (2.0 * dimension * dimension * dimension) / (elapsedMilliseconds.count() / 1000.0);
        outputFile << flops << "         ";

        transpose(B, rows);
		auto startTime1 = chrono::high_resolution_clock::now();
        multySequentialUnrolledMultiThread(A, B, C_My, dimension, 8);  // Use 8 threads
        auto endTime1 = chrono::high_resolution_clock::now();

        chrono::duration<double, milli> elapsedMilliseconds1 = endTime1 - startTime1;
        double flops1 = (2.0 * dimension * dimension * dimension) / (elapsedMilliseconds1.count() / 1000.0);
        outputFile << flops1 << "  ";

        double residual = calculateResidual(C_Blas, C_My);
        outputFile << residual << "  ";
        bool withinTolerance = (residual < 1e-6);

        outputFile << withinTolerance << "\n";

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

            //Output the result calculated by matrix-matrix-multiply Thread solution 
            cout << "    Thread Results: \n";
            printMatrix(C_My, rows);

            cout << endl;
            cout << "    Time of Thread: " << elapsedMilliseconds1.count();
            cout << endl;
            cout << "    FLOPs of Thread: " << flops1 << endl;
            cout << "    Residual of Thread: " << residual << endl;

            // Check if the residual is within the tolerance
            if (residual < 1e-6) {
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


            // Output the result calculated by cblas_dgemv of BLAS
            cout << "    BLAS Results: \n";
            cout << "    Time of BLAS: " << elapsedMilliseconds.count();
            cout << endl;
            cout << "    FLOPs of BLAS: " << flops << endl << endl;

            //Output the result calculated by matrix-matrix-multiply Thread solution 
            cout << "    Thread Results: \n";
            cout << endl;
            cout << "    Time of Thread: " << elapsedMilliseconds1.count();
            cout << endl;
            cout << "    FLOPs of Thread: " << flops1 << endl;
            cout << "    Residual of Thread: " << residual << endl;

            // Check if the residual is within the tolerance
            if (residual < 1e-6) {
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
