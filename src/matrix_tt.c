#include "../include/paralleltt.h"
#include <math.h>

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <cblas.h>
#include <string.h>
#include <lapacke.h>

#define min(a,b) ((a)>(b)?(b):(a))
#define abs(a) ((a)>(0)?(a):(-a))

#ifndef HEAD
#define HEAD (int) 0
#endif

matrix_tt* matrix_tt_init(const int m, const int n)
{
    matrix_tt* A = (matrix_tt*) malloc(sizeof(matrix_tt));
    A->m = m;
    A->n = n;
    A->transpose = 0;
    A->offset = 0;
    A->lda = m;
    A->X_size = (long) m*n;
    A->X = (double*) malloc(A->X_size*sizeof(double));

    return A;
}

void matrix_tt_wrap_update(matrix_tt* A, int m, int n, double* X)
{
    A->m = m;
    A->n = n;
    A->transpose = 0;
    A->offset = 0;
    A->lda = m;
    A->X_size = (long) m*n;
    A->X = X;
}

matrix_tt* matrix_tt_wrap(int m, int n, double* X)
{
    matrix_tt* A = (matrix_tt*) malloc(sizeof(matrix_tt));
    A->m = m;
    A->n = n;
    A->transpose = 0;
    A->offset = 0;
    A->lda = m;
    A->X_size = (long) m*n;
    A->X = X;
    return A;
}

matrix_tt* matrix_tt_copy(const matrix_tt* A)
{
    matrix_tt* B = (matrix_tt*) malloc(sizeof(matrix_tt));
    int m = A->m; int n = A->n; int X_size = A->X_size;
    B->m = m;
    B->n = n;
    B->transpose = A->transpose;
    B->offset = A->offset;
    B->lda = A->lda;
    B->X_size = X_size;
    B->X = (double*) malloc(X_size * sizeof(double));
    memcpy(B->X, A->X, X_size * sizeof(double));

    return B;
}

// Copy from matrix A to matrix B
void matrix_tt_copy_data(matrix_tt* B, const matrix_tt* A)
{
    int BT = (B->transpose == 0) ? 0 : 1;
    int AT = (A->transpose == 0) ? 0 : 1;
    if (BT == AT){
        if ((B->m != A->m) || (B->n != A->n)){
            printf("matrix_tt_copy_data: B (%d x %d) is not the same size as A (%d x %d)\n", B->n, B->m, A->n, A->m);
            return;
        }
        for (int jj = 0; jj < B->n; ++jj){
            memcpy((B->X) + (B->offset) + (B->lda)*jj, (A->X) + (A->offset) + jj*(A->lda), (B->m)*sizeof(double));
        }
    }
    else{
        if ((B->m != A->n) || (B->n != A->m)){
            printf("matrix_tt_copy_data: B (%d x %d, transpose = %d) is not the same size as A (%d x %d, transpose = %d)\n",
                    B->n, B->m, B->transpose, A->n, A->m, A->transpose);
            return;
        }
        for (int jj = 0; jj < B->n; ++jj){
            double* BX = (B->X) + (B->offset) + (B->lda)*jj;
            double* AX = (A->X) + (A->offset) + jj;
            for (int ii = 0; ii < B->m; ++ii){
                BX[ii] = AX[ii*(A->lda)];
            }
        }
    }
}

matrix_tt* submatrix_copy(const matrix_tt* A)
{
    matrix_tt* B = (matrix_tt*) malloc(sizeof(matrix_tt));
    int m = A->m; int n = A->n; int X_size = m*n;
    B->m = m;
    B->n = n;
    B->transpose = A->transpose;
    B->offset = 0;
    B->lda = m;
    B->X_size = X_size;
    B->X = (double*) malloc(X_size * sizeof(double));
    matrix_tt_copy_data(B, A);

    return(B);
}

// NOTE: reshape and submatrix will not work well together.
void matrix_tt_reshape(int m, int n, matrix_tt* A)
{
    long new_size = (long) m*n;
    if (new_size > A->X_size){
        printf("Reshape failed: m*n=%ld is too large for X_size=%ld\n", (long) new_size, A->X_size);
    }
    else{
        A->m = m;
        A->n = n;
        A->offset = 0;
        A->lda = m;
    }
}

// In matlab notation, returns A[ii1:ii2-1, jj1:jj2-1]
matrix_tt* submatrix(const matrix_tt* A, int ii1, int ii2, int jj1, int jj2)
{
    int mA = A->m; int nA = A->n; int lda = A->lda;
    if ((ii1 < 0) || (ii2 < ii1) || (mA < ii2) || (jj1 < 0) || (jj2 < jj1) || (nA < jj2)){
        printf("Cannot take the A[%d:%d,%d:%d] of a %d by %d matrix (Matlab notation)\n", ii1, ii2-1, jj1, jj2-1, mA, nA);
    }
    matrix_tt* B = (matrix_tt*) malloc(sizeof(matrix_tt));

    B->m = ii2 - ii1;
    B->n = jj2 - jj1;
    B->transpose = A->transpose;
    B->offset = ii1 + jj1*lda;
    B->lda = lda;
    B->X_size = A->X_size;
    B->X = A->X + A->offset;

    return B;
}

// Updates the indices of the submatrix A
void submatrix_update(matrix_tt* A, int ii1, int ii2, int jj1, int jj2)
{
    A->m = ii2 - ii1;
    A->n = jj2 - jj1;
    A->offset = ii1 + jj1*(A->lda);
}


// Get an element of the matrix
double matrix_tt_element(const matrix_tt* A, int ii, int jj)
{
    return A->X[A->offset + ii + (A->lda)*jj];
}

void matrix_tt_free(matrix_tt* A)
{
    free(A->X); A->X = NULL;
    free(A);
    return;
}

void matrix_tt_print(const matrix_tt* A, int just_the_matrix)
{
    int m = A->m; int n = A->n; int T = A->transpose;
    int offset = A->offset; int lda = A->lda;

    if (!just_the_matrix){
        printf("Size of matrix: m=%d, n=%d\n", m, n);
        printf("Transpose=%d, offset=%d, lda=%d\n", T, offset, lda);
        printf("X_size=%ld\n", A->X_size);
        printf("X");
        if (T){ printf("^T"); }
        printf(" is \n");
    }

    int mloop = T ? m : n;
    int nloop = T ? n : m;
    printf("[");
    for (int jj = 0; jj < nloop; ++jj){
        for (int ii = 0; ii < mloop; ++ii){
            if (!T){ printf("%f",A->X[lda*ii + jj + offset]); }
            else{    printf("%f",A->X[lda*jj + ii + offset]); }
            if (ii < mloop-1){ printf(", "); }
        }

        if (jj == nloop - 1){
            printf("]\n");
        }
        else{
            printf(",\n ");
        }
    }
}


double frobenius_norm(matrix_tt* A){
    double norm = 0;
    if (A->m == A->lda){
        norm = cblas_dnrm2(A->m * A->n, A->offset + A->X, 1);
    }
    else{
        for (int ii = 0; ii < A->n; ++ii){
            double tmp = cblas_dnrm2(A->m, A->offset + (A->lda)*ii + A->X, 1);
            norm = sqrt(norm*norm + tmp*tmp);
        }
    }
    return norm;
}



// Performs the operation C = alpha*A*B + beta*C, where A and B are appropriately transposed
int matrix_tt_dgemm(const matrix_tt* A, const matrix_tt* B, matrix_tt* C, const double alpha, const double beta)
{
    int mA = A->m; int nA = A->n;
    int mB = B->m; int nB = B->n;
    int m_check = C->m; int n_check = C->n; int k_check;

    int m; int n; int k;
    CBLAS_TRANSPOSE TransA;
    CBLAS_TRANSPOSE TransB;
    if (A->transpose == 0){
        TransA = CblasNoTrans;
        m = mA;
        k = nA;
    }
    else{
        TransA = CblasTrans;
        m = nA;
        k = mA;
    }


    if (B->transpose == 0){
        TransB = CblasNoTrans;
        k_check = mB;
        n = nB;
    }
    else{
        TransB = CblasTrans;
        k_check = nB;
        n = mB;
    }

    if ((m != m_check) || (n != n_check) || (k != k_check)){
        printf("Dimensions of input arrays do not match:\n");
        printf("A: m=%d, n=%d, tranpose=%d\n",mA,nA,A->transpose);
        printf("B: m=%d, n=%d, tranpose=%d\n",mB,nB,B->transpose);
        printf("C: m=%d, n=%d, tranpose=%d\n",m_check,n_check,C->transpose);
        return 1;
    }

    C->transpose = 0;

    cblas_dgemm(CblasColMajor, TransA, TransB,
                m, n, k,
                alpha,
                A->X + A->offset, A->lda,
                B->X + B->offset, B->lda,
                beta,
                C->X + C->offset, C->lda);

    return 0;
}

// Solves the least squares problem Ax = b
// At the end, it just copies the result into x. This probably isn't the most efficient way to do this
void matrix_tt_dgels(matrix_tt* x, matrix_tt* A, const matrix_tt* b)
{
    if (x->transpose != 0){
        printf("matrix_tt_least_squares: x cannot be transposed for dgels\n");
        return;
    }

    if (b->transpose != 0){
        printf("matrix_tt_least_squares: b cannot be transposed for dgels\n");
        return;
    }

    int nrhs = x->n;
    char TransA;
    int m_check;
    int n_check;


    if (A->transpose == 0){
        TransA = 'N';
        m_check = A->m;
        n_check = A->n;
    }
    else{
        TransA = 'T';
        m_check = A->n;
        n_check = A->m;
    }

    if (n_check != x->m){
        printf("right index of A (%d) does not match left index of x (%d)\n", n_check, x->m);
    }
    if (m_check != b->m){
        printf("left index of A (%d) does not match left index of b (%d)\n", m_check, b->m);
    }
    if (x->n != b->n){
        printf("right index of x (%d) does not match right index of b (%d)\n", x->n, b->n);
    }

    int m = A->m;
    int n = A->n;

    int info = LAPACKE_dgels(LAPACK_COL_MAJOR, TransA, m, n, nrhs, A->X + A->offset, A->lda, b->X + b->offset, b->lda);

    matrix_tt* result = submatrix(b, 0, x->m, 0, x->n);
    matrix_tt_copy_data(x, result);
    free(result);
}


// Input:
//   Q - the matrix you want to take the QR
// Output:
//   Q - the Q matrix of QR
//   R - (input is NULL) ? nothing : the R matrix of QR

// NOTE: Currently assumes R and Q are not transposed
int matrix_tt_truncated_qr(matrix_tt* Q, matrix_tt* R, int r)
{
    int Qm = Q->m; int Qn = Q->n; double* QX = Q->X;
    int Qtranspose = Q->transpose; long Qoffset = Q->offset; int Qlda = Q->lda;

    lapack_int info;


    if (Qtranspose == 0){
        if (r > Qn)
        {
            printf("ERROR: (matrix_tt_truncated_qr) r = %d is too large for Q->n = %d \n", r, Qn);
            return 1;
        }
        // The QR step
        double* tau = (double*) malloc(sizeof(double)*min(Qm,Qn));
        info = LAPACKE_dgeqrf(LAPACK_COL_MAJOR, Qm, Qn, QX + Qoffset, Qlda, tau);

        if (R != NULL){
            int Rm = R->m; int Rn = R->n; double* RX = R->X;
            int Rtranspose = R->transpose; long Roffset = R->offset; int Rlda = R->lda;

            if ((Rm != r) || (Rn != r)){
                printf("ERROR: (matrix_tt_truncated_qr) R->m = %d or R->n = %d is not equal to r = %d\n", Rm, Rn, r);
                return 1;
            }
            if (Rtranspose != 0){
                printf("ERROR: (matrix_tt_truncated_qr) Only defined for R->transpose = 0\n");
                return 1;
            }

            for (int jj = 0; jj < r; ++jj){
                int r_col_length = (Qm < jj+1) ? (Qm) : jj+1;
                memcpy(RX + Roffset + jj*Rlda, QX + Qoffset + jj*Qlda, r_col_length*sizeof(double));
                memset(RX + Roffset + jj*Rlda + r_col_length, 0, (r-r_col_length)*sizeof(double));
            }
        }

        if (r > Qm){
            LAPACKE_dorgqr(LAPACK_COL_MAJOR, Qm, Qm, Qm, QX + Qoffset, Qlda, tau);
            matrix_tt* Q_sub = submatrix(Q, 0, Qm, Qm, r);
            matrix_tt_fill_zeros(Q_sub);
            free(Q_sub);
        }
        else{
            LAPACKE_dorgqr(LAPACK_COL_MAJOR, Qm, r, r, QX + Qoffset, Qlda, tau);
        }

        Q->n = r;

        free(tau);       tau = NULL;
    }
    else{
        if (r > Qm)
        {
            printf("ERROR: (matrix_tt_truncated_qr) r = %d is too large for Q->n = %d \n", r, Qn);
            return 1;
        }
        // The LQ step
        printf("Performing LQ\n");
        double* tau = (double*) malloc(sizeof(double)*min(Qm,Qn));
        info = LAPACKE_dgelqf(LAPACK_COL_MAJOR, Qm, Qn, QX + Qoffset, Qlda, tau);
        printf("\nRight after LQ, Q = \n");
        matrix_tt_print(Q, 1);

        if (R != NULL){
            int Rm = R->m; int Rn = R->n; double* RX = R->X;
            int Rtranspose = R->transpose; long Roffset = R->offset; int Rlda = R->lda;

            if ((Rm != r) || (Rn != r)){
                printf("ERROR: (matrix_tt_truncated_qr) R->m = %d or R->n = %d is not equal to r = %d\n", Rm, Rn, r);
                return 1;
            }
            if (Rtranspose == 0){
                printf("ERROR: (matrix_tt_truncated_qr) Only defined for R->transpose = Q->transpose\n");
                return 1;
            }

            int ncols = (r > Qm) ? Qm : r;
            for (int jj = 0; jj < r; ++jj){
                if (jj < Qm){
                    memcpy(RX + Roffset + jj*Rlda + jj, QX + Qoffset + jj*Qlda + jj, (Qm - jj)*sizeof(double));
                    memset(RX + Roffset + jj*Rlda, 0, jj*sizeof(double));
                }
            }

        }

        if (r > Qn){
            printf("Doing first dorglq\n");
            LAPACKE_dorglq(LAPACK_COL_MAJOR, Qn, Qn, Qn, QX + Qoffset, Qlda, tau);
            matrix_tt* Q_sub = submatrix(Q, Qn, r, 0, Qn);
            matrix_tt_fill_zeros(Q_sub);
            free(Q_sub);
        }
        else{
            printf("Doing second dorglq\n");
            LAPACKE_dorglq(LAPACK_COL_MAJOR, Qm, Qn, Qm, QX + Qoffset, Qlda, tau);
        }

        Q->m = r;

        free(tau);       tau = NULL;
    }

    return 0;
}/**/

void matrix_tt_group_reduce(MPI_Comm comm, int rank, matrix_tt* A, matrix_tt* buf, int head, int* group_ranks, int nranks)
{
    int in_the_group = 0;
    for (int ii = 0; ii < nranks; ++ii){
        if (rank == group_ranks[ii]){
            in_the_group = 1;
        }
    }

    if (in_the_group){
        int buf_assigned = 0;
        if (!buf){
            buf_assigned = 1;
            buf = matrix_tt_init(A->m, A->n);
        }

        if ((buf->m < A->m) || (buf->n < A->n)){
            printf("buf (%d x %d) must be larger than A (%d x %d)\n", buf->m, buf->n, A->m, A->n);
        }

        int send = -1;
        int recv = -1;
        while (nranks > 1){
            int half_nranks = (nranks + 1) / 2;
            for (int ii = 0; ii < half_nranks; ++ii){
                if (ii + half_nranks >= nranks){
                    recv = group_ranks[ii];
                    send = -1;
                }
                else if (group_ranks[ii + half_nranks] == head){
                    recv = head;
                    send = group_ranks[ii];
                }
                else{
                    recv = group_ranks[ii];
                    send = group_ranks[ii + half_nranks];
                }

                if (rank == send){
                    matrix_tt_send(comm, A, recv);
                }
                else if ((rank == recv) && (send != -1)){
                    matrix_tt_recv(comm, buf, send);
                    for (int jj = 0; jj < A->n; ++jj){
//                        MPI_Recv(buf->X + buf->offset, A->m, MPI_DOUBLE, send, 0, comm, MPI_STATUS_IGNORE);
                        long A_offset = A->offset + jj * (A->lda);
                        long buf_offset = buf->offset +  jj * (buf->lda);
                        for (int kk = 0; kk < A->m; ++kk){
                            A->X[A_offset + kk] = A->X[A_offset + kk] + buf->X[buf_offset + kk];
                        }
                    }
                }
                group_ranks[ii] = recv;
            }
            nranks = half_nranks;
        }

        if (buf_assigned){
            matrix_tt_free(buf);
        }
    }

}


// buf just has to be size A->lda x 1 or more
void matrix_tt_reduce(MPI_Comm comm, int rank, matrix_tt* A, matrix_tt* buf, int head)
{
    if (buf == NULL){
        matrix_tt* buf = matrix_tt_init(A->m, 1);
        matrix_tt_reduce(comm, rank, A, buf, head);
        matrix_tt_free(buf);
    }
    else{
        int size;
        MPI_Comm_size(comm, &size);

        if ((head < 0) || (head >= size)){
            printf("matrix_tt_reduce: head = %d is not a valid rank for size = %d\n", head, size);
        }

        if (buf->lda < A->m){
            printf("buf->lda = %d must be larger than A->m = %d\n", buf->lda, A->m);
        }

        int* activated_ranks = (int*) calloc(size, sizeof(int));
        for (int ii = 0; ii < size; ++ii){
            activated_ranks[ii] = ii;
        }
        int send = -1;
        int recv = -1;
        while (size > 1){
            int half_size = (size + 1) / 2;
            for (int ii = 0; ii < half_size; ++ii){
                if (ii + half_size >= size){
                    recv = activated_ranks[ii];
                    send = -1;
                }
                else if (activated_ranks[ii + half_size] == head){
                    recv = head;
                    send = activated_ranks[ii];
                }
                else{
                    recv = activated_ranks[ii];
                    send = activated_ranks[ii + half_size];
                }

                if (rank == send){
                    for (int jj = 0; jj < A->n; ++jj){
                        MPI_Send(A->X + A->offset + A->lda * jj, A->m, MPI_DOUBLE, recv, 0, comm);
                    }
                }
                else if ((rank == recv) && (send != -1)){
                    for (int jj = 0; jj < A->n; ++jj){
                        MPI_Recv(buf->X + buf->offset, A->m, MPI_DOUBLE, send, 0, comm, MPI_STATUS_IGNORE);
                        long A_offset = A->offset + jj * (A->lda);
                        for (int kk = 0; kk < A->m; ++kk){
                            A->X[A_offset + kk] = A->X[A_offset + kk] + buf->X[buf->offset + kk];
                        }
                    }
                }
                activated_ranks[ii] = recv;
            }
            size = half_size;
        }
        free(activated_ranks);
    }
}

void matrix_tt_allreduce(MPI_Comm comm, matrix_tt* A)
{
    for (int jj = 0; jj < A->n; ++jj){
        MPI_Allreduce(MPI_IN_PLACE, A->X + A->offset + jj*(A->lda), A->m, MPI_DOUBLE, MPI_SUM, comm);
    }
}

void matrix_tt_broadcast(MPI_Comm comm, matrix_tt* buf)
{
    int count = buf->m * buf->n;
    MPI_Bcast(buf->X, count, MPI_DOUBLE, HEAD, comm);
}

void matrix_tt_send(MPI_Comm comm, matrix_tt* buf, int dest){
    MPI_Datatype buf_type;
    MPI_Type_vector(buf->n, buf->m, buf->lda, MPI_DOUBLE, &buf_type);
    MPI_Type_commit(&buf_type);
    MPI_Send(buf->X + buf->offset, 1, buf_type, dest, 0, comm);
    MPI_Type_free(&buf_type);
}

void matrix_tt_recv(MPI_Comm comm, matrix_tt* buf, int source){
    MPI_Datatype buf_type;
    MPI_Type_vector(buf->n, buf->m, buf->lda, MPI_DOUBLE, &buf_type);
    MPI_Type_commit(&buf_type);
    MPI_Recv(buf->X + buf->offset, 1, buf_type, source, 0, comm, MPI_STATUS_IGNORE);
    MPI_Type_free(&buf_type);
}

// Column khatri-rao product C = A x B
void khatri_rao(const matrix_tt* restrict A, const matrix_tt* restrict B, matrix_tt* restrict C){
    int n = A->n;
    int m_A = A->m;

    int n_B = B->n;
    int m_B = B->m;

    int n_C = C->n;
    int m_C = C->m;

    if ((n != n_B) || (n != n_C) || (m_A*m_B != m_C)){
        printf("khatri_rao: the sizes dont work!\n");
        printf("A is %d x %d\n", m_A, n);
        printf("B is %d x %d\n", m_B, n_B);
        printf("C is %d x %d\n", m_C, n_C);
    }

    for (int kk = 0; kk < n; ++kk){
        for (int ii = 0; ii < m_B; ++ii){
            double Bii = B->X[kk*m_B + ii];
            for (int jj = 0; jj < m_A; ++jj){
                C->X[kk*m_C + ii*m_B + jj] = Bii * A->X[kk*m_A + jj];
            }
        }
    }
}

// Recursively KR multiply the list of matrices A
void list_khatri_rao(int d_kr, matrix_tt** A, matrix_tt* Omega){
    if (d_kr == 1){
        for (int ii = 0; ii < A[0]->X_size; ++ii){
            Omega->X[ii] = A[0]->X[ii];
        }
        return;
    }

    if (d_kr == 2){
        khatri_rao(A[0], A[1], Omega);
    }
    else{
        int m_B = 1;
        int n_B = Omega->n;
        for (int ii = 0; ii < d_kr-1; ++ii){
            m_B = m_B * A[ii]->m;
        }

        matrix_tt* B = matrix_tt_init(m_B, n_B);
        list_khatri_rao(d_kr - 1, A, B);
        khatri_rao(B, A[d_kr - 1], Omega);
        matrix_tt_free(B); B = NULL;
    }

}


void matrix_tt_dlarnv(matrix_tt* A){
    int r1 = rand()%4096, r2 = rand()%4096, r3 = rand()%4096, r4 = rand()%4096;
    int iseed[4] = {r1, r2, r3, r4+(r4%2 == 0?1:0)};
    LAPACKE_dlarnv(3, iseed, (A->n) * (A->m), A->X);
}


void matrix_tt_fill_zeros(matrix_tt* A)
{
    for (int jj = 0; jj < A->n; ++jj){
        memset(A->X + A->offset + jj*A->lda, 0, A->m * sizeof(double));
    }
}

// Takes a matrix in row major and transforms it to column major
void row_to_col_major(const matrix_tt* mat_row, matrix_tt* mat_col)
{
    int m = mat_col->m;
    int n = mat_col->n;
    if ((m != mat_row->n) || (n != mat_row->m)){
        printf("The dimensions don't work out! mat_row should have the dimensions of mat_col transposed\n");
        printf("mat_row->m = %d, mat_row->n = %d\n", mat_row->m, mat_row->n);
        printf("mat_col->m = %d, mat_col->n = %d\n", mat_col->m, mat_col->n);
    }

    for (int ii = 0; ii < m; ++ii){
        for (int jj = 0; jj < n; ++jj){
            mat_col->X[jj * (mat_col->lda) + ii + mat_col->offset] = mat_row->X[ii * (mat_row->lda) + jj + mat_row->offset];
        }
    }
}
