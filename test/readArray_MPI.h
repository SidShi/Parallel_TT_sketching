#ifndef READARRAY_MPI_H
#define READARRAY_MPI_H

#include <mpi.h>
#include <gperftools/heap-profiler.h>
#include "../include/paralleltt.h"

// Gets a tensor created from a tensor train with i.i.d. Gaussian entries
//MPI_tensor* get_gaussian(const int d, const int* n, const int* r);

// Gets a tensor created from a Hilbert entries A_{ijk...} = 1/(1+i+j+k+...)
tensor_train* get_gaussian_tt(const int d, const int* n, const int* r, const unsigned int seed);


// Hilbert functions
typedef struct p_hilbert {
    int d;           // dimension of tensor
    int* n;          // tensor size of each dimension
    int* n_stream;   // Temporary memory for streaming
    long* n_prod;
//    int*
} p_hilbert;
void* p_hilbert_init(int d, int* n);
void p_hilbert_free(void* parameters);
void f_hilbert(double* restrict X, int* ind1, int* ind2, const void* parameters);

// Calculates the sum of Gaussian bumps.
typedef struct p_gaussian_bumps {
    int d;  // dimension of tensor
    int* n; // tensor size of each dimensions

    int M;        // Number of bumps
    double gamma; // Bump length scale

    double* region;  // 2 x d array of bounds for the rectangular region ([xa, xb, ya, yb, za, zb,...])
    double* centers; // d x M array of bump centers ([x1 y1 z1 ..., x2 y2 z2 ..., ...])

    double* x_ii; // d array that stores the current location at ii
    int* n_stream; // d array that stores the size of the stream
    long* n_prod;  // d array that stores information to get tensor index
} p_gaussian_bumps;
void* p_gaussian_bumps_init(int d, int* n, int M, double gamma, double* region, double* centers);
void* unit_random_p_gaussian_bumps_init(int d, int* n, int M, double gamma, int seed);
void p_gaussian_bumps_free(void* parameters);
void f_gaussian_bumps(double* restrict X, int* ind1, int* ind2, const void* parameters);


//void f_randtens(MPI_tensor* ten, long N1, long N2, const void* parameters);
//
//void f_gaussian(MPI_tensor* ten, long N1, long N2, const void* parameters);


// arithmetic functions (ten[ii] = ii)

typedef struct p_arithmetic {
    int d;
    int* n;
} p_arithmetic;

void* p_arithmetic_init(int d, int* n);
void p_arithmetic_free(void* parameters);
void f_arithmetic(double* restrict X, int* ind1, int* ind2, const void* parameters);


typedef struct p_tt {
    tensor_train* tt;
} p_tt;

void* p_tt_init(tensor_train* tt);
void p_tt_free(void* parameters);
void f_tt(double* restrict X, int* ind1, int* ind2, const void* parameters);

double tt_error(tensor_train* tt, MPI_tensor* ten);

#endif
