#include "../include/paralleltt.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <cblas.h>
#include <lapacke.h>
#include <unistd.h>
#include <mpi.h>
#include <gperftools/heap-profiler.h>

#define NUM_THREAD 1
#define min(a,b) ((a)>(b)?(b):(a))
#define VERBOSE ((int) 1)

#ifndef HEAD
#define HEAD ((int) 0)
#endif

#define BUF ((int) 2)




tensor_train* TT_init(const int d, const int* restrict n)
{
    tensor_train* sim = (tensor_train*) malloc(sizeof(tensor_train));
    sim->d = d;
    sim->n = (int*) malloc(sizeof(int)*d);
    sim->r = (int*) malloc(sizeof(int)*(d+1));
    sim->trains = (double**) malloc(sizeof(double*)*d);

    sim->r[0] = 1;
    for (int j = 0; j < d; ++j) {
        sim->n[j] = n[j];

        if (j < d-1) {
            sim->r[j+1] = n[j];
        }
    }
    sim->r[d] = 1;

    for (int j = 0; j < d; ++j) {
        sim->trains[j] = (double*) malloc(sizeof(double)*(sim->r[j])*(sim->n[j])*(sim->r[j+1]));
    }

    return sim;
}

tensor_train* TT_init_rank(const int d, const int* restrict n, const int* restrict r)
{
    tensor_train* sim = (tensor_train*) malloc(sizeof(tensor_train));
    sim->d = d;
    sim->n = (int*) malloc(sizeof(int)*d);
    sim->r = (int*) malloc(sizeof(int)*(d+1));
    sim->trains = (double**) malloc(sizeof(double*)*d);

    for (int j = 0; j < d; ++j) {
        sim->n[j] = n[j];
    }
    for (int j = 0; j < d+1; ++j) {
        sim->r[j] = r[j];
    }

    for (int j = 0; j < d; ++j) {
        sim->trains[j] = (double*) malloc(sizeof(double)*(sim->r[j])*(sim->n[j])*(sim->r[j+1]));
    }

    return sim;
}

tensor_train* TT_copy(tensor_train* X)
{
    int d = X->d;
    int* n = X->n;
    int* r = X->r;
    tensor_train* Y = TT_init_rank(d, n, r);
    for (int j = 0; j < X->d; ++j) {
        int N = r[j] * n[j] * r[j+1];
        memcpy(Y->trains[j], X->trains[j], N * sizeof(double));
    }

    return Y;
}

void TT_free(tensor_train* sim)
{
    for (int j = 0; j < sim->d; ++j) {
        free(sim->trains[j]); sim->trains[j] = NULL;
    }
    free(sim->trains); sim->trains = NULL;
    free(sim->r);      sim->r = NULL;
    free(sim->n);      sim->n = NULL;
    free(sim);
}

void TT_print(tensor_train* tt)
{

    int d = tt->d;
    int* n = tt->n;
    int* r = tt->r;
    double** trains = tt->trains;
    printf("Printing the tensor train with d = %d\n",d);

    for (int ii = 0; ii < d; ++ii){
        printf("\nTensor %d/d (r[%d] = %d, n[%d]=%d, r[%d]=%d):\n",ii,ii,r[ii],ii,n[ii],ii+1,r[ii+1]);
        matrix* A = matrix_wrap(r[ii], n[ii] * r[ii+1], trains[ii]);
        matrix_print(A, 1);
        free(A); A = NULL;
    }
}

// Multiplies out the tensor train X, to recover the original tensor Y
// NOTE: This isn't the most memory efficient implementation. However, I think it's probably good enough for something
//       that doesn't actually go in the main timing loop (probably).
//void TT_to_tensor(const tensor_train* tt, MPI_tensor* ten, long N1, long N2)
//{
//    int d = tt->d;
//    int* n = tt->n;
//    int* r = tt->r;
//
//    double** trains = tt->trains;
//
//    if (N1 == N2){ return; } // ten is empty, just do nothing
//
//    long* sizes = flattening_sizes(ten, d - 1);
//    long iid_min = floor( N1 / sizes[0]);
//    long iid_max = floor( (N2-1) / sizes[0]);
//    long nd = iid_max - iid_min + 1;
//
//
//    // Allocate the matrices to their maximum sizes
//    matrix* C = matrix_init((int) sizes[0], nd);
//    matrix* B = matrix_init((int) sizes[0], nd);
//
//    // Set their sizes for the first iteration
//    B->m = r[d-1]; // B in r[d-1] x n[d-1]
//    C->m = r[d-2]* n[d-2]; // C in r[d-2] n[d-2] x n[d-1]
//
//    // Set initial B value to the final tensor train, only with the required entries
//    memcpy(B->X, trains[d-1] + (iid_min*r[d-1]), sizeof(double) * r[d-1] * nd);
//
//    // THIS LOOP GOES BACKWARDS!!!
//    for (int ii = d-2; ii >= 0; --ii)
//    {
//        matrix* A = matrix_wrap(r[ii] * n[ii], r[ii+1], trains[ii]);
//        matrix_reshape(r[ii+1], (B->m / r[ii+1]) * B->n, B);
//        matrix_reshape(A->m, B->n, C);
//
//        matrix_dgemm(A, B, C, 1.0, 0.0);
//
//        // Switch the pointers for the next loop
//        matrix* tmp = B;
//        B = C;
//        C = tmp;
//
//        free(A); A = NULL;
//    }
//
//    // Copy over the multiplied out tensor to ten
//    memcpy(ten->X, B->X + (N1 - iid_min*sizes[0]), (N2 - N1) * sizeof(double));
//
//    matrix_free(B); B = NULL;
//    matrix_free(C); C = NULL;
//    free(sizes);    sizes = NULL;
//}

void tt_broadcast(MPI_Comm comm, tensor_train* tt){
    for (int ii = 0; ii < tt->d; ++ii){
        matrix* A = matrix_wrap((tt->r[ii]) * (tt->n[ii]) * (tt->r[ii+1]), 1, tt->trains[ii]);
        matrix_broadcast(comm, A);
        free(A);
    }
}


double get_compression(const tensor_train* tt)
{
    long N = 1;
    for (int ii = 0; ii < tt->d; ++ii)
    {
        N = N * tt->n[ii];
    }

    long Ntt = 0;
    for (int ii = 0; ii < tt->d; ++ii)
    {
        Ntt = Ntt + ((tt->r[ii])*(tt->n[ii])*(tt->r[ii+1]));
    }

    double compression = (double) Ntt/N;
    return compression;
}

