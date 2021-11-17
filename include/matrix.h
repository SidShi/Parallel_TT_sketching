/* matrix.h:
 /   This file contains everything related to constructing and working with the matrix data struct. This struct
 /   contains a matrix X, its sizes m and n, and its rank r.
*/

#include <mpi.h>
#include <gperftools/heap-profiler.h>

#ifndef MATRIX_H
#define MATRIX_H

typedef struct matrix {
    int m;
    int n;
    int transpose;

    long offset;
    int lda;

    long X_size; // Do not change unless you reassign X
    double* X;
} matrix;


matrix* matrix_init(const int m, const int n);
void matrix_wrap_update(matrix* A, int m, int n, double* X);
matrix* matrix_wrap(int m, int n, double* X);
matrix* matrix_copy(const matrix* A);
void matrix_free(matrix* A);
//void matrix_transpose(matrix* A);

void matrix_print(const matrix* A, int just_the_matrix);

void matrix_reshape(int m, int n, matrix* A);
void matrix_copy_data(matrix* B, const matrix* A);
matrix* submatrix_copy(const matrix* A);
matrix* submatrix(const matrix* A, int ii1, int ii2, int jj1, int jj2);
void submatrix_update(matrix* A, int ii1, int ii2, int jj1, int jj2);
double matrix_element(const matrix* A, int ii, int jj);

double frobenius_norm(matrix* A);
int matrix_dgemm(const matrix* A, const matrix* B, matrix* C, const double alpha, const double beta);
void matrix_dgels(matrix* x, matrix* A, const matrix* b);
void matrix_fill_zeros(matrix* A);
int matrix_truncated_qr(matrix* Q, matrix* R, int r);
void matrix_reduce(MPI_Comm comm, int rank, matrix* A, matrix* buf, int head);
void matrix_group_reduce(MPI_Comm comm, int rank, matrix* A, matrix* buf, int head, int* group_ranks, int nranks);
void matrix_allreduce(MPI_Comm comm, matrix* A);
void matrix_broadcast(MPI_Comm comm, matrix* buff);

void matrix_send(MPI_Comm comm, matrix* buf, int dest);
void matrix_recv(MPI_Comm comm, matrix* buf, int source);

void matrix_dlarnv(matrix* A);

void find_idx1(int* lidx, const int* n, const int d, const int N, const int index);
void find_idx2(int* lidx, const int* n, const int d, const long N, const long index);
void fast_kr_multiply(const matrix* X, matrix** Psi, matrix* Y, int d_kr, int* m_kr, int r,
                      int tot, int idx1, int idx2, double beta);

void row_to_col_major(const matrix* mat_row, matrix* mat_col);

#endif
