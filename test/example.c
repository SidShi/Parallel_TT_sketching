#include "../include/paralleltt.h"
#include "./readArray_MPI.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


void PSTT2_test(int d, const int* n, const int* r, int* nps, void (*f_ten)(double* restrict, int*, int*, const void*), void* parameters)
{
    // Allocate tensor
    MPI_tensor* ten = MPI_tensor_init(d, n, nps, MPI_COMM_WORLD, f_ten, parameters);

    // Allocate tensor train
    tensor_train* tt = TT_init_rank(d, n, r);
    int mid = -1;

    // Find tensor train via PSTT2
    MPI_Barrier(ten->comm);
    double t0 = MPI_Wtime();
    PSTT2(tt, ten, mid);
    double t1 = MPI_Wtime();

    // Get and print relative error
    double err = tt_error(tt, ten);
    if (ten->rank == 0){
        printf("\nERROR = %e\nTime taken by 0th core = %e\n", err, t1-t0);
    }

    TT_free(tt);
    MPI_tensor_free(ten);
}

void SSTT_test(int d, const int* n, const int* r, int* nps, void (*f_ten)(double* restrict, int*, int*, const void*), void* parameters)
{
    MPI_tensor* ten = MPI_tensor_init(d, n, nps, MPI_COMM_WORLD, f_ten, parameters);
    tensor_train* tt = TT_init_rank(d, n, r);

    MPI_Barrier(ten->comm);
    double t0 = MPI_Wtime();
    SSTT(tt, ten);
    double t1 = MPI_Wtime();

    // Re-allocate tensor, as SSTT deallocated the original tensor!
    ten = MPI_tensor_init(d, n, nps, MPI_COMM_WORLD, f_ten, parameters);

    double err = tt_error(tt, ten);

    if (ten->rank == 0){
        printf("\nERROR = %e\nTime taken by 0th core = %e\n", err, t1-t0);
    }

    TT_free(tt);
    MPI_tensor_free(ten);
}

void PSTT2_onepass_test(int d, const int* n, const int* r, int* nps, void (*f_ten)(double* restrict, int*, int*, const void*), void* parameters)
{
    MPI_tensor* ten = MPI_tensor_init(d, n, nps, MPI_COMM_WORLD, f_ten, parameters);
    tensor_train* tt = TT_init_rank(d, n, r);
    int mid = -1;

    MPI_Barrier(ten->comm);
    double t0 = MPI_Wtime();
    PSTT2_onepass(tt, ten, mid);
    double t1 = MPI_Wtime();

    double err = tt_error(tt, ten);

    if (ten->rank == 0){
        printf("\nERROR = %e\nTime taken by 0th core = %e\n", err, t1-t0);
    }

    TT_free(tt);
    MPI_tensor_free(ten);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    // Test to run:
    //   1: PSTT2
    //   2: PSTT2-onepass
    //   3: SSTT
    int test_num = 1;

    // Evaluation function
    //   1: Arithmetic
    //   2: Hilbert
    //   3: Gaussian bumps
    int tensortype = 2;

    // Input Tensor arguments
    int d = 5;   // Dimension
    int n0 = 30; // Sizes
    int r0 = 10; // Ranks

    int* n = (int*) malloc(d * sizeof(int));
    int* r = (int*) malloc((d+1) * sizeof(int));
    for (int ii = 0; ii < d; ++ii){
        n[ii] = n0 ;
        r[ii] = r0;
    }
    r[0] = 1; r[d] = 1;

    // Subtensor partition sizes
    int* nps = (int*) malloc(d * sizeof(int));
    for (int ii = 0; ii < d; ++ii){
        nps[ii] = 1;
    }
    nps[0] = 4;
    nps[d-1] = 4;

    // Get the tensor evaluation function and parameters
    void (*f_ten)(double* restrict, int*, int*, const void*);
    void* parameters;
    tensor_train* tt = NULL;

    if (tensortype == 1){ // Arithmetic tensor evaluation X[i] = i
        f_ten = &f_arithmetic;
        parameters = p_arithmetic_init(d, n);
    }
    else if (tensortype == 2){ // Hilbert tensor evaluation X[i1, i2, ...] = 1/(1 + i1 + i2 + ...)
        f_ten = &f_hilbert;
        parameters = p_hilbert_init(d, n);
    }
    else if (tensortype == 3){ // Gaussian bumps evaluation sum_{j=0}^M-1 exp(-gamma ( (x-xj)^2 + (y-yj)^2  + ... ) )
        int M = 10;           // Number of bumps
        double gamma = 10.0;  // sharpness of bumps
        int seed = 168234591;
        parameters = unit_random_p_gaussian_bumps_init(d, n, M, gamma, seed);
        f_ten = &f_gaussian_bumps;
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0){
        printf("Starting tensor decomposition, test_num=%d\n", test_num);
    }

    // Perform the test
    if (test_num == 1){
        PSTT2_test(d, n, r, nps, f_ten, parameters);
    }
    else if(test_num == 2){
        PSTT2_onepass_test(d, n, r, nps, f_ten, parameters);
    }
    else if(test_num == 3){
        SSTT_test(d, n, r, nps, f_ten, parameters);
    }

    // Free memory
    if(tensortype == 1){
        p_arithmetic_free(parameters); parameters = NULL;
    }
    else if(tensortype == 2){
        p_hilbert_free(parameters); parameters = NULL;
    }
    else if(tensortype == 3){
        p_gaussian_bumps_free(parameters); parameters = NULL;
    }


    free(n);
    free(r);
    free(nps);
    if (rank == 0){
        printf("Ending test\n");
    }
    MPI_Finalize();

    return 0;
}
