#include "readArray_MPI.h"
#include "tensor.h"
#include "matrix.h"
#include "sketch.h"
#include "PSTT.h"
#include "SSTT.h"
#include "VTime.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <gperftools/heap-profiler.h>




void stream_test(int d, const int* n, int* nps, void (*f_ten)(double* restrict, int*, int*, const void*), void* parameters)
{
    MPI_tensor* ten = MPI_tensor_init(d, n, nps, MPI_COMM_WORLD, f_ten, parameters);
    stream(ten, 0);
    MPI_tensor_print(ten, 2);
}

void sketch_test(int d, const int* n, const int* r, int* nps, void (*f_ten)(double* restrict, int*, int*, const void*), void* parameters)
{
    MPI_tensor* ten = MPI_tensor_init(d, n, nps, MPI_COMM_WORLD, f_ten, parameters);
    sketch* s;
    int flattening = 2;
    int iscol = 1;
    int buf = 2;

    sketch** sketches = (sketch**) malloc((d-1) * sizeof(sketch*));
    for (int ii = 0; ii < d-1; ++ii){
        sketches[ii] = sketch_init(ten, ii+1, r[ii+1], buf, iscol);
    }
    printf("Initialized sketches\n");
    int i1 = 1; int i2 = 2;
//    multi_perform_sketch(sketches+i1, i2-i1);
    printf("Sketched the sketches\n");
    for(int ii = i1; ii < i2; ++ii){
        sketch_qr(sketches[ii]);
        sketch_print(sketches[ii]);
        free(sketches[ii]); sketches[ii] = NULL;
    }
    free(sketches);

//    double* A = (double*) calloc(4,sizeof(double));
//    int rank;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    A[0] = 1.0;
//    A[1] = 2.0;
//
//    MPI_Gather(A, 2, MPI_DOUBLE, A, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//
//    for (int ii = 0; ii < 4; ++ii){
//        printf("r%d A[%d] = %f\n", rank, ii, A[ii]);
//    }


}

void matrix_qr_test()
{
    int m = 6;
    int n = 3;
//    int r = (m>n) ? n : m;
    int r = 3;
    matrix* A_holder = matrix_init(2*m,2*n);
    matrix* A = submatrix(A_holder, 1, 1+m, 1, 1+n);
//    matrix* A = submatrix(A_holder, 1, 1, 1, 1+n);
//    matrix* A = matrix_init(m, n);
    matrix_dlarnv(A);
    for (int ii = 0; ii < 4*m*n; ++ii){
        A->X[ii] = ii;
    }
    matrix* Q = matrix_copy(A);
    matrix* QT = matrix_init(A->n, A->m);
    QT->transpose = 1;
    matrix_copy_data(QT, A);
    matrix* R_big = matrix_init(5*r, r);
    matrix* R = submatrix(R_big, 3, r+3, 0,r);
    matrix* RT = matrix_init(r,r);
    RT->transpose = 1;

    matrix_truncated_qr(Q,R,r);
    matrix_truncated_qr(QT, RT, r);

    printf("A = \n");
    matrix_print(A, 1);

    printf("\nQ = \n");
    matrix_print(Q, 1);

    printf("\nR = \n");
    matrix_print(R, 1);

    printf("\nQT = \n");
    matrix_print(QT, 1);

    printf("\nRT = \n");
    matrix_print(RT, 1);

//    printf("\nR_big = \n");
//    matrix_print(R_big, 1);

//    matrix* QR = matrix_init(Q->m, r);
//    matrix_dgemm(Q, R, QR, 1.0, 0.0);
//    printf("\nQ*R = \n");
//    matrix_print(QR, 1);


}

void MPI_group_test()
{
//    matrix* A = matrix_init(2,2);
//    double* X = A->X;
//    X[0] = 0;
//    X[1] = 1;
//    X[2] = 2;
//    X[3] = 3;
//
//    MPI_Group world_group;
//    MPI_Group color_group = MPI_GROUP_NULL;
//    MPI_Comm color_comm = MPI_COMM_NULL;
//
//    int* group_ranks = (int*) calloc(2, sizeof(int));
//    group_ranks[0] = 0;
//    group_ranks[1] = 2;
//    int kk = 2;
//
//    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
//    MPI_Group_incl(world_group, kk, group_ranks, &color_group);
//    MPI_Comm_create_group(MPI_COMM_WORLD, color_group, 0, &color_comm);
//
//    int rank;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    if ((rank == 0) || (rank == 2)){
//        int group_rank;
//        MPI_Comm_rank(color_comm, &group_rank);
//        printf("group_rank = %d\n", group_rank);
//        matrix_reduce(color_comm, group_rank, A, 1);
//    }
//    matrix_reduce(MPI_COMM_WORLD, rank, A, 2);
//    matrix_print(A,1);
}

void PSTT2_test(int d, const int* n, const int* r, int* nps, void (*f_ten)(double* restrict, int*, int*, const void*), void* parameters)
{
    MPI_tensor* ten = MPI_tensor_init(d, n, nps, MPI_COMM_WORLD, f_ten, parameters);
    tensor_train* tt = TT_init_rank(d, n, r);
    int mid = -1;
    VTime* time = PSTT2(tt, ten, mid);
//
    double err = tt_error(tt, ten);

    if (ten->rank == 0){
//        TT_print(tt);
//        int N_check = 1;
//        for (int ii = 0; ii < d-1; ++ii){
//            N_check = N_check*n[ii];
//        }
//        matrix* check = matrix_init(N_check, n[d-1]);
//        void* tt_parameters = p_tt_init(tt);
//
//        int* ind1 = (int*) calloc(d, sizeof(int));
//        int* ind2 = (int*) calloc(d, sizeof(int));
//        for (int ii = 0; ii < d; ++ii){
//            ind1[ii] = 0;
//            ind2[ii] = n[ii];
//        }
//
//        f_tt(check->X, ind1, ind2, tt_parameters);
//        p_tt_free(tt_parameters);

//        printf("\nMULTIPLIED TENSOR TRAIN\n");
//        matrix_print(check, 1);
//        matrix_free(check); check = NULL;
//
//        free(ind1); ind1 = NULL;
//        free(ind2); ind2 = NULL;

        printf("\n ERROR = %e\n", err);
    }

    TT_free(tt);
    MPI_tensor_free(ten);

    VTime_print(time);
    VTime_free(time);
}

void VTime_test()
{
    VTime* tm = VTime_init(2);
//    usleep(1000);
    VTime_break(tm, 0, "first break");
//    usleep(1000);
    VTime_break(tm, 0, "second break");
//    usleep(1000);
    VTime_finalize(tm);

    VTime_gathered* tms = VTime_gather(tm, MPI_COMM_WORLD);
    VTime_print(tm);
    VTime_free(tm);
}

void SSTT_test(int d, const int* n, const int* r, int* nps, void (*f_ten)(double* restrict, int*, int*, const void*), void* parameters)
{
    MPI_tensor* ten = MPI_tensor_init(d, n, nps, MPI_COMM_WORLD, f_ten, parameters);
    tensor_train* tt = TT_init_rank(d, n, r);
    int mid = -1;
    VTime* time = SSTT(tt, ten);
    ten = MPI_tensor_init(d, n, nps, MPI_COMM_WORLD, f_ten, parameters);

    double err = tt_error(tt, ten);

    if (ten->rank == 0){
//        TT_print(tt);
//        int N_check = 1;
//        for (int ii = 0; ii < d-1; ++ii){
//            N_check = N_check*n[ii];
//        }
//        matrix* check = matrix_init(N_check, n[d-1]);
//        void* tt_parameters = p_tt_init(tt);
//
//        int* ind1 = (int*) calloc(d, sizeof(int));
//        int* ind2 = (int*) calloc(d, sizeof(int));
//        for (int ii = 0; ii < d; ++ii){
//            ind1[ii] = 0;
//            ind2[ii] = n[ii];
//        }
//
//        f_tt(check->X, ind1, ind2, tt_parameters);
//        p_tt_free(tt_parameters);
//        free(ind1);
//        free(ind2);

//        printf("\nMULTIPLIED TENSOR TRAIN\n");
//        matrix_print(check, 1);
//        matrix_free(check);

        printf("\n ERROR = %e\n", err);
    }

    TT_free(tt);
    MPI_tensor_free(ten);

    VTime_print(time);
    VTime_free(time);

}

void dgels_test()
{
    double A[6] = {0, 1, 2, 3, 4, 5};
    matrix* A_mat = matrix_wrap(2,3,A);
    A_mat->transpose = 1;
    double* X = (double*) malloc(2*3*sizeof(double));
    matrix* X_mat = matrix_wrap(2,3,X);
    double b[9] = {0, 1, 2, 3, 4, 5, 0, 0, 1};
    matrix* b_mat = matrix_wrap(3,3,b);

    printf("\nA = \n");
    matrix_print(A_mat,1);
    printf("\nb = \n");
    matrix_print(b_mat,1);

    matrix_dgels(X_mat, A_mat, b_mat);
    printf("\nx = \n");
    matrix_print(X_mat,1);
}

void reduce_test()
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int m = 4;
    int n = 4;
    int head = 2;
    matrix* A = matrix_init(m, n);
    for (int ii = 0; ii < m*n; ++ii){
        A->X[ii] = ii;
    }
    if (rank == head){
        printf("\nr%d A before = \n", rank);
        matrix_print(A, 1);
    }
    matrix* buf = matrix_init(m, 1);

    matrix_reduce(MPI_COMM_WORLD, rank, A, buf, head);

    if (rank == head){
        printf("\nr%d A after = \n", rank);
        matrix_print(A, 1);
    }

}

void PSTT2_onepass_test(int d, const int* n, const int* r, int* nps, void (*f_ten)(double* restrict, int*, int*, const void*), void* parameters)
{
    MPI_tensor* ten = MPI_tensor_init(d, n, nps, MPI_COMM_WORLD, f_ten, parameters);
    tensor_train* tt = TT_init_rank(d, n, r);
    int mid = -1;
    VTime* time = PSTT2_onepass(tt, ten, mid);

    double err = tt_error(tt, ten);

    if (ten->rank == 0){
//        TT_print(tt);
//        int N_check = 1;
//        for (int ii = 0; ii < d-1; ++ii){
//            N_check = N_check*n[ii];
//        }
//        matrix* check = matrix_init(N_check, n[d-1]);
//        void* tt_parameters = p_tt_init(tt);
//
//        int* ind1 = (int*) calloc(d, sizeof(int));
//        int* ind2 = (int*) calloc(d, sizeof(int));
//        for (int ii = 0; ii < d; ++ii){
//            ind1[ii] = 0;
//            ind2[ii] = n[ii];
//        }
//
//        f_tt(check->X, ind1, ind2, tt_parameters);
//        p_tt_free(tt_parameters);

//        printf("\nMULTIPLIED TENSOR TRAIN\n");
//        matrix_print(check, 1);
//        matrix_free(check); check = NULL;

//        free(ind1); ind1 = NULL;
//        free(ind2); ind2 = NULL;

        printf("\n ERROR = %e\n", err);
    }

    TT_free(tt);
    MPI_tensor_free(ten);

    VTime_print(time);
    VTime_free(time);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int d = 5;
    int n0 = 5;
    int r0 = 4;

    int* n = (int*) malloc(d * sizeof(int));
    int* r = (int*) malloc((d+1) * sizeof(int));
    for (int ii = 0; ii < d; ++ii){
        n[ii] = n0 ;
        r[ii] = r0;
    }
    r[0] = 1; r[d] = 1;

    int* nps = (int*) malloc(d * sizeof(int));
    for (int ii = 0; ii < d; ++ii){
        nps[ii] = 2;
    }
    nps[0] = 1;
//    nps[1] = 1;
//    nps[d-1] = 2;
//    nps[0] = 2;

    int test_num = 7;
    int tensortype = 1;

    void (*f_ten)(double* restrict, int*, int*, const void*);
    void* parameters;
    tensor_train* tt = NULL;


    if (tensortype == 0){ // Arithmetic, for testing functions
        f_ten = &f_arithmetic;
        parameters = p_arithmetic_init(d, n);
    }
    else if (tensortype == 1){
        f_ten = &f_hilbert;
        parameters = p_hilbert_init(d, n);
    }
    else if (tensortype == 2){
        int M = 400;
        double gamma = 10.0;
        int seed = 168234591;
        parameters = unit_random_p_gaussian_bumps_init(d, n, M, gamma, seed);
        f_ten = &f_gaussian_bumps;
    }
    else if (tensortype == 3){ // Known Gaussian Bumps function for testing
        if (d != 3){
            printf("tensortype 3 not defined for d != 3\n");
            return 1;
        }
        int M = 2;
        int gamma = 1;
        double* region = calloc(6, sizeof(double));
        for (int ii = 0; ii < 3; ++ii){
            region[2*ii] = -1;
            region[2*ii + 1] = 1;
        }
        double* centers = calloc(d*M, sizeof(double));
        centers[0] = 0.1;
        centers[1] = 0.1;
        centers[2] = 0.1;
        parameters = p_gaussian_bumps_init(d, n, M, gamma, region, centers);
        f_ten = &f_gaussian_bumps;

        free(region);
        free(centers);
    }
    else if (tensortype == 4){ // For testing tensor_train stream
        f_ten = &f_arithmetic;
        parameters = p_arithmetic_init(d,n);

        MPI_tensor* ten = MPI_tensor_init(d, n, nps, MPI_COMM_WORLD, f_ten, parameters);
        tt = TT_init_rank(d, n, r);

        int mid = -1;
        PSTT2(tt, ten, mid);

        p_arithmetic_free(parameters);
        MPI_tensor_free(ten);

        parameters = p_tt_init(tt);
        f_ten = &f_tt;
    }
//    void* unit_random_p_gaussian_bumps_init(int d, int* n, int M, double gamma, int seed);



    if (test_num == 0){
        stream_test(d, n, nps, f_ten, parameters);
    }
    else if(test_num == 1){
        sketch_test(d, n, r, nps, f_ten, parameters);
    }
    else if(test_num == 2){
        matrix_qr_test();
    }
    else if(test_num == 3){
        PSTT2_test(d, n, r, nps, f_ten, parameters);
    }
    else if(test_num == 4){
        VTime_test();
    }
    else if(test_num == 5){
        SSTT_test(d, n, r, nps, f_ten, parameters);
    }
    else if (test_num == 6){
        dgels_test();
    }
    else if(test_num == 7){
        PSTT2_onepass_test(d, n, r, nps, f_ten, parameters);
    }
    else if(test_num == 8){
        reduce_test();
    }

    if(tensortype == 0){
        p_arithmetic_free(parameters); parameters = NULL;
    }
    else if(tensortype == 1){
        p_hilbert_free(parameters); parameters = NULL;
    }
    else if((tensortype == 2) || (tensortype == 3)){
        p_gaussian_bumps_free(parameters); parameters = NULL;
    }


    free(n);
    free(r);
    free(nps);

    MPI_Finalize();

    return 0;
}
