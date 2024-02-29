/* Complile Steps: running using command line of windows system
 *
 * Step1: install Openblas, pay attention to adding path in the system variables
 * Step2: open cmd, navigate to the file holding matrixVectorMutiply_S.cpp
 * Step2: input g++ matrixVectorMutiply_S.cpp -o sequential.exe -L/path/to/include -L/path/to/lib -lopenblas
 * -----  For example: g++ matrixVectorMutiply_S.cpp -o sequential.exe -I"D:/xzAssignment/software/OpenBLAS-0.3.26-x64/include" -L"D:/software/OpenBLAS-0.3.26-x64/lib" -lopenblas
 * Step3: input ¡°sequential.exe¡±
 *
*/
/*  * Introduction: matrixVectorMultiply_S.cpp contains the implementation of Sequentail solution.
 * 1. Tool part: Provide tool functions for main() part
 *  1) Sequential Solution Tool
 *   ---- multySequential() is to do Matrix-Vector multiplication for Sequential solution.
 *   ---- generateSequentialResult() to generate results for Sequential solution,
 *        including flops of solution, residual of sequential and BLAS, consuming time.
 *  2) BLAS Tool
 *   ---- flattenMatrix() to tranfer 2D matrix to 1D matrix.
 *  3) Print Out Tool
 *   ---- printMatrix() to print out 2D matrix into screen.
 *   ---- printVector() to print out vector into screen.
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
 *  3) Part 3: Sequential Solution
 *   ---- Get Y (= A * X), flaps1, residual1, elapsedMilliseconds1 by calling tool
 *        function generateSequentialResult().
 *        In generateSequentialResult(), it calls tool function multySequential() to implement A * X.
 *        In generateSequentialResult(), it calculates multySequential() implementation to get FLOPs of Sequential solution.
 *        In generateSequentialResult(), it calculates residual1 of Sequential solution and BLAS solution.
 *   ---- Check the resitual1 tolerance (= 1e-15), within tolerance is true, or not is false.
 *  4) Part 4: Partial Results Ouput
 *   ---- Print out Marix A, Vector X, Vector Z,flaps of BLAS, Vector Y, consuming time of sequencital solution
 *   flaps of sequentiaol solution, residual, residual tolerance
 *
 * 3. Others:
 *  1) outputFile:
 *     It will be generated under the current running program folder automatically.
 *     Every line of outputFile includes: FLAPs of BLAS, FLAPs of Sequential, Resitual of Sequential,
 *     Resitual of Sequential Tolerance.
 *  2) Number of Test Cases
 *     It will generate cases from starting by 10, up to 10000, increasement is 10.
 *  3) Output of screen:
 *     Because with the increase of dimensions, it is impossible to output all the results,
 *     the output of screen just shows dimension <= 60 result detail.
 *
 */


#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cblas.h>

using namespace std;

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

//============================ Sequential Solution Tool ============================= 
// Function to do Matrix-Vector multiplication for Sequential solution
void multySequential(const vector<vector<double>>& A, const vector<double>& X, vector<double>& Y1, int dimension) {

    for (int i = 0; i < dimension; ++i) {
        Y1[i] = 0;
        for (int j = 0; j < dimension; ++j) {
            Y1[i] += A[i][j] * X[j];
        }
    }
}

// Function to generate results for Sequential solution, 
// including flops of solution, residual of sequential and BLAS, consuming time
void generateSequentialResult(const vector<vector<double>>& A, const vector<double>& X, vector<double>& Y1, vector<double>& Z, int dimension, double& flops, double& residual, chrono::duration<double, milli>& elapsedMilliseconds1) {

    // Measure the start time, Time (Milliseconds)
    auto startTime = chrono::high_resolution_clock::now();

    // Perform matrix-vector multiplication
    multySequential(A, X, Y1, dimension);

    // Measure the end time
    auto endTime = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> elapsedMilliseconds = endTime - startTime;
    elapsedMilliseconds1 = elapsedMilliseconds;

    // Calculate FLOPS
    double flops1 = (2.0 * dimension * dimension) / (elapsedMilliseconds1.count() / 1000.0);
    flops = flops1;


    // Get residual of matrix-vector-multiply Sequential solution and BLAS
    double residual1 = 0.0;
    residual1 = calculateResidual(Z, Y1);
    residual = residual1;

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


int main() {

    // Open the file for writing the results
    // File name will be "matrix_vector_sequential_4.txt" when running
    // in my 4-core computer and using Visual Studio complier; And it also will be 
    // "matrix_vector_sequential_8.txt" when running on other 8-core computer and 
    // using Visual Studio complier.
    ofstream outputFile("matrix_vector_sequential_g++_4.txt");

    // Check if the file is open
    if (!outputFile.is_open()) {
        cerr << "Unable to open output file.\n";
        return 0;
    }


    // Declear vector Y1, Z
    vector<vector<double>> A; // For matrix
    vector<double> X;         // For Vector
    vector<double> Y;  // For storing sequential solution results
    vector<double> Z;  // For storing BLAS results

    const double tolerance = 1e-6; // For residual tolerance checking
    bool withinTolerance = false; // For record sequential residual tolerance status

    for (int dimension = 10; dimension <= 10000; dimension += 10) {

        int rows = dimension;
        int cols = dimension;

        //==========================  Part 1 =========================== 
        //==================== Generate test case ======================
        
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
        Y.resize(rows, 0.0);

        // Initialize results vector Y  
        Z.resize(rows, 0.0);


        //===========================  Part 2 =========================== 
        //======================== BLAS Solution =======================

        // Calculate FLOPs, residual of matrix-vector-multiply thread solution and BLAS

        // Flatten matrix A for cblas_dgemv
        std::vector<double> flattenedA = flattenMatrix(A);

        // Measure the start time, Time (Milliseconds)
        auto startTime = chrono::high_resolution_clock::now();

        // Using CBLAS for vector-matrix multiplication: Z = alpha * A * X + beta * Z
        cblas_dgemv(CblasRowMajor, CblasNoTrans, rows, cols, 1, flattenedA.data(), cols, X.data(), 1, 0, Z.data(), 1);


        // Measure the end time
        auto endTime = chrono::high_resolution_clock::now();

        // Calculate time consuming of BLAS
        chrono::duration<double, milli> elapsedMilliseconds = endTime - startTime;

        // Calculate FLOPs of BLAS
        double flops = (2.0 * rows * cols) / (elapsedMilliseconds.count() / 1000.0);

        // Write FLOPS to the output file
        outputFile << flops << "         ";


        //===========================  Part 3 =========================== 
        //===================== Solution Implementation =====================

        // Declare parameters to collect results
        double flops1 = 0;
        double residual1 = 0;
        chrono::duration<double, milli> elapsedMilliseconds1 = std::chrono::milliseconds(1);

        generateSequentialResult(A, X, Y, Z, dimension, flops1, residual1, elapsedMilliseconds1);

        // Write FLOPS to the output file
        outputFile << flops1 << "  ";

        // Write residual of  matrix-vector-multiply sequential solution and BLAS into the outputFile
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

            //Output the result calculated by matrix-vector-multiply sequential solution 
            cout << "    Sequential Results: ";
            for (int i = 0; i < rows; ++i) {
                cout << " " << Y[i];
            }
            cout << endl;
            cout << "    Time of Sequential: " << elapsedMilliseconds1.count();
            cout << endl;
            cout << "    FLOPs of Sequential: " << flops1 << endl;
            cout << "    Residual of Sequential: " << residual1 << endl;

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
