#include "readArray_MPI.h"
#include "tensor.h"
#include "matrix.h"
#include "tt.h"
#include "sketch.h"
//#include "tt.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <lapacke.h>
#include <time.h>
#include <math.h>
#include <gperftools/heap-profiler.h>
// #include <string.h>

void* p_hilbert_init(int d, int* n){
    p_hilbert* parameters = (p_hilbert*) malloc(sizeof(p_hilbert));
    parameters->d = d;
    parameters->n = (int*) malloc(d*sizeof(int));
    for (int ii = 0; ii < d; ++ii){
        parameters->n[ii] = n[ii];
    }
    parameters->n_stream = (int*) calloc(d, sizeof(int));
    parameters->n_prod = (long*) malloc(d * sizeof(long));
    return (void*) parameters;
}


void p_hilbert_free(void* parameters){
    p_hilbert* p_casted = (p_hilbert*) parameters;
    free(p_casted->n); p_casted->n = NULL;
    free(p_casted->n_stream); p_casted->n_stream = NULL;
    free(p_casted->n_prod); p_casted->n_prod = NULL;
    free(p_casted);
}

void f_hilbert(double* restrict X, int* ind1, int* ind2, const void* parameters)
{
    p_hilbert* p_casted = (p_hilbert*) parameters;
    int d = p_casted->d;
    int* n = p_casted->n;
    int* n_stream = p_casted->n_stream;
    long* n_prod = p_casted->n_prod;


    long N_stream = 1;
    for (int ii = 0; ii < d; ++ii){
        n_stream[ii] = ind2[ii] - ind1[ii];
        N_stream = N_stream*n_stream[ii];
    }

    n_prod[0] = 1;
    for (int ii = 1; ii < d; ++ii){
        n_prod[ii] = (long) n_prod[ii-1] * n_stream[ii-1];
    }

    int bias = 1;
    for (int jj = 0; jj < d; ++jj){
        bias = bias + ind1[jj];
    }

    for (long ii = 0; ii < N_stream; ++ii){
        int xinv = bias;
		long tmp = ii;
		for (int jj = d-1; jj >= 0; jj--){
			xinv += tmp/n_prod[jj];
			tmp = tmp % n_prod[jj];
		}

        X[ii] = (double) 1/xinv;
    }
}

void* p_gaussian_bumps_init(int d, int* n, int M, double gamma, double* region, double* centers)
{
    p_gaussian_bumps* parameters = (p_gaussian_bumps*) malloc(sizeof(p_gaussian_bumps));
    parameters->d = d;
    parameters->n = (int*) calloc(d, sizeof(int));
    for (int ii = 0; ii < d; ++ii){
        parameters->n[ii] = n[ii];
    }

    parameters->M = M;
    parameters->gamma = gamma;

    parameters->region = (double*) calloc(2*d, sizeof(double));
    for (int ii = 0; ii < 2*d; ++ii){
        parameters->region[ii] = region[ii];
    }
    parameters->centers = (double*) calloc(d*M, sizeof(double));
    for (int ii = 0; ii < d*M; ++ii){
        parameters->centers[ii] = centers[ii];
    }

//    parameters->ten_ind_ii = (int*) calloc(d, sizeof(int));
    parameters->x_ii = (double*) calloc(d, sizeof(double));
    parameters->n_stream = (int*) calloc(d, sizeof(double));
    parameters->n_prod = (long*) calloc(d, sizeof(long));

    return (void*) parameters;
}

void* unit_random_p_gaussian_bumps_init(int d, int* n, int M, double gamma, int seed)
{
    p_gaussian_bumps* parameters = (p_gaussian_bumps*) malloc(sizeof(p_gaussian_bumps));
    parameters->d = d;
    parameters->n = (int*) calloc(d, sizeof(int));
    for (int ii = 0; ii < d; ++ii){
        parameters->n[ii] = n[ii];
    }

    parameters->M = M;
    parameters->gamma = gamma;

    parameters->region = (double*) calloc(2*d, sizeof(double));
    for (int ii = 0; ii < d; ++ii){
        parameters->region[2*ii] = -1.0;
        parameters->region[2*ii+1] = 1.0;
    }

    parameters->centers = (double*) calloc(d*M, sizeof(double));
    srand(seed);
    int r1 = rand()%4096, r2 = rand()%4096, r3 = rand()%4096, r4 = rand()%4096;
    int iseed[4] = {r1, r2, r3, r4+(r4%2 == 0?1:0)};
    LAPACKE_dlarnv(2, iseed, d*M, parameters->centers);

    parameters->x_ii = (double*) calloc(d, sizeof(double));
    parameters->n_stream = (int*) calloc(d, sizeof(double));
    parameters->n_prod = (long*) calloc(d, sizeof(long));

    return (void*) parameters;
}

void p_gaussian_bumps_free(void* parameters)
{
    p_gaussian_bumps* p_casted = (p_gaussian_bumps*) parameters;

    free(p_casted->n);          p_casted->n = NULL;
    free(p_casted->region);     p_casted->region = NULL;
    free(p_casted->centers);    p_casted->centers = NULL;
    free(p_casted->x_ii);       p_casted->x_ii = NULL;
    free(p_casted->n_stream);   p_casted->n_stream = NULL;
    free(p_casted->n_prod);     p_casted->n_prod = NULL;
    free(p_casted);
}

void f_gaussian_bumps(double* restrict X, int* ind1, int* ind2, const void* parameters)
{
    p_gaussian_bumps* p_casted = (p_gaussian_bumps*) parameters;
    int d = p_casted->d;
    int* n = p_casted->n;
    int M = p_casted->M;
    double gamma = p_casted->gamma;
    double* region = p_casted->region;
    double* centers = p_casted->centers;
    double* x_ii = p_casted->x_ii;
    int* n_stream = p_casted->n_stream;
    long* n_prod = p_casted->n_prod;

    long N_stream = 1;
    for (int ii = 0; ii < d; ++ii){
        n_stream[ii] = ind2[ii] - ind1[ii];
        N_stream = N_stream*n_stream[ii];
    }

    n_prod[0] = 1;
    for (int ii = 1; ii < d; ++ii){
        n_prod[ii] = (long) n_prod[ii-1] * n_stream[ii-1];
    }

    for (long ii = 0; ii < N_stream; ++ii){
        long tmp = ii;
        for (int jj = d-1; jj >= 0; jj--){
            int ind_jj = tmp/n_prod[jj];
            x_ii[jj] = region[2*jj] + (ind_jj + ind1[jj]) * (region[2*jj + 1] - region[2*jj]) / (n[jj] - 1);
			tmp = tmp % n_prod[jj];
		}

        X[ii] = 0;
        for (int kk = 0; kk < M; ++kk){
            double exponent = 0;
            for (int jj = 0; jj < d; ++jj){
                exponent += (x_ii[jj] - centers[kk*d + jj]) * (x_ii[jj] - centers[kk*d + jj]);
            }
            X[ii] += exp(-gamma * exponent);
        }
    }
}

void* p_arithmetic_init(int d, int* n)
{
    p_arithmetic* parameters = (p_arithmetic*) malloc(sizeof(p_arithmetic));
    parameters->d = d;
    parameters->n = (int*) malloc(d*sizeof(int));
    for (int ii = 0; ii < d; ++ii){
        parameters->n[ii] = n[ii];
    }
    return (void*) parameters;
}

void p_arithmetic_free(void* parameters)
{
    p_arithmetic* p_casted = (p_arithmetic*) parameters;
    free(p_casted->n); p_casted->n = NULL;
    free(p_casted);
}


void f_arithmetic(double* restrict X, int* ind1, int* ind2, const void* parameters)
{
    p_arithmetic* p_casted = (p_arithmetic*) parameters;
    int d = p_casted->d;
    int* n = p_casted->n;

    int* n_stream = (int*) calloc(d, sizeof(int));
    long N_stream = 1;
    for (int ii = 0; ii < d; ++ii){
        n_stream[ii] = ind2[ii] - ind1[ii];
        N_stream = N_stream*n_stream[ii];
    }

    int* ind_ii = (int*) calloc(d, sizeof(int));
    for (int ii = 0; ii < N_stream; ++ii){
        to_tensor_ind(ind_ii, ii, n_stream, d);
        for (int jj = 0; jj < d; ++jj){
            ind_ii[jj] = ind_ii[jj] + ind1[jj];
        }
        X[ii] = 1 + to_vec_ind(ind_ii, n, d);
//        X[ii] = X[ii] + (0.000000001 / X[ii]);
    }

    free(n_stream);
    free(ind_ii);
}



void* p_tt_init(tensor_train* tt)
{
    p_tt* parameters = (p_tt*) malloc(sizeof(p_tt));
    parameters->tt = tt;
    return (void*) parameters;
}

// Doesn't actually free the tensor train. You gotta do that yourself
void p_tt_free(void* parameters)
{
    p_tt* p_casted = (p_tt*) parameters;
    p_casted->tt = NULL;
    free(p_casted);
}

void f_tt(double* restrict X, int* ind1, int* ind2, const void* parameters)
{
    p_tt* p_casted = (p_tt*) parameters;
    tensor_train* tt = p_casted->tt;
    int d = tt->d;
    int* n = tt->n;
    int* r = tt->r;

//    matrix** train_mats = (matrix**) calloc(d, sizeof(matrix*));
    matrix** train_submats = (matrix**) calloc(d, sizeof(matrix*));
    for (int ii = 0; ii < d; ++ii){
        matrix* train_mat_ii = matrix_wrap(r[ii]*n[ii], r[ii+1], tt->trains[ii]);
        train_submats[ii] = submatrix(train_mat_ii, ind1[ii]*r[ii], ind2[ii]*r[ii], 0, r[ii+1]);
        free(train_mat_ii);
    }

    matrix* mult = submatrix_copy(train_submats[d-1]);
//    printf("train_submats[d-1] = (%d x %d)", train_submats[d-1]->m, train_submats[d-1]->n);
//    printf(", reshaping to (%d x %d)\n", r[d-1], ind2[d-1] - ind1[d-1]);
    matrix_reshape(r[d-1], ind2[d-1] - ind1[d-1], mult);


//    matrix_print(mult, 0);
    for (int jj = d-2; jj >= 1; --jj){
//        printf("train_submats[jj]->m = %d, mult->n = %d\n",train_submats[jj]->m, mult->n);
        matrix* new_mult = matrix_init(train_submats[jj]->m, mult->n);
        matrix_dgemm(train_submats[jj], mult, new_mult, 1.0, 0.0);
        matrix_reshape(r[jj], (new_mult->n) * (new_mult->m) / r[jj], new_mult);
        matrix_free(mult); mult = new_mult;

//        matrix_print(mult, 0);
    }

    long X_m = 1;
    for (int ii = 1; ii < d; ++ii){
        X_m = X_m*(ind2[ii] - ind1[ii]);
    }

    matrix* X_mat = matrix_wrap(ind2[0] - ind1[0], X_m, X);
    matrix_dgemm(train_submats[0], mult, X_mat, 1.0, 0.0);

    for (int ii = 0; ii < d; ++ii){
        free(train_submats[ii]); train_submats[ii] = NULL;
    }

    free(train_submats); train_submats = NULL;
    matrix_free(mult);   mult = NULL;
    free(X_mat);         X_mat = NULL;
}


double tt_error(tensor_train* tt, MPI_tensor* ten)
{
    MPI_Comm comm = ten->comm;
    tt_broadcast(comm, tt);


    void* parameters_tt = p_tt_init(tt);
    MPI_tensor* ten_tt = MPI_tensor_init(ten->d, ten->n, ten->nps, ten->comm, &f_tt, parameters_tt);
    int rank = ten_tt->rank;
    int* schedule_rank = ten_tt->schedule[rank];

    double true_norm_squared = 0;
    double diff_norm_squared = 0;

    flattening_info* fi = flattening_info_init(ten, 0, 1, 0);
    for (int ii = 0; ii < ten->n_schedule; ++ii){
        int block = schedule_rank[ii];

        if (block != -1){
            stream(ten, block);
            stream(ten_tt, block);

            flattening_info_update(fi, ten, block);
            long N = fi->t_N;

            for (int jj = 0; jj < N; ++jj){
                ten_tt->X[jj] = ten_tt->X[jj] - ten->X[jj];
            }

            matrix* mat_true = matrix_wrap(N, 1, ten->X);
            double tmp = frobenius_norm(mat_true);
            true_norm_squared = true_norm_squared + tmp*tmp;

            matrix* mat_diff = matrix_wrap(N, 1, ten_tt->X);
            tmp = frobenius_norm(mat_diff);
            diff_norm_squared = diff_norm_squared + tmp*tmp;
//            printf("r%d ii%d true_norm_squared = %e, diff_norm_squared = %e\n", rank, ii, true_norm_squared, diff_norm_squared);

            free(mat_true); mat_true = NULL;
            free(mat_diff); mat_diff = NULL;
        }
    }
    int head = 0;

    double true_norm_reduced;
    double diff_norm_reduced;
    MPI_Reduce(&true_norm_squared, &true_norm_reduced, 1, MPI_DOUBLE, MPI_SUM, head, comm);
    true_norm_reduced = sqrt(true_norm_reduced);

    MPI_Reduce(&diff_norm_squared, &diff_norm_reduced, 1, MPI_DOUBLE, MPI_SUM, head, comm);
    diff_norm_reduced = sqrt(diff_norm_reduced);

    double rel_err = diff_norm_reduced/true_norm_reduced;



    p_tt_free(parameters_tt); parameters_tt = NULL;
    flattening_info_free(fi); fi = NULL;
    MPI_tensor_free(ten_tt); ten_tt = NULL;

    return rel_err;
}