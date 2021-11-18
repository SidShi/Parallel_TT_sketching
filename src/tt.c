#include "../include/paralleltt.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <cblas.h>
#include <lapacke.h>
#include <unistd.h>
#include <mpi.h>

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
        matrix_tt* A = matrix_tt_wrap(r[ii], n[ii] * r[ii+1], trains[ii]);
        matrix_tt_print(A, 1);
        free(A); A = NULL;
    }
}

void tt_broadcast(MPI_Comm comm, tensor_train* tt){
    for (int ii = 0; ii < tt->d; ++ii){
        matrix_tt* A = matrix_tt_wrap((tt->r[ii]) * (tt->n[ii]) * (tt->r[ii+1]), 1, tt->trains[ii]);
        matrix_tt_broadcast(comm, A);
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

