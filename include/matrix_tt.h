/* matrix_tt.h:
 /   This file contains everything related to constructing and working with the matrix_tt data struct. This struct
 /   contains a matrix X, its sizes m and n, and its rank r.
*/

#include <mpi.h>

#ifndef MATRIX_TT_H
#define MATRIX_TT_H

typedef struct matrix_tt {
    int m;         // column length
    int n;         // row length
    int transpose; // 0 -> not transposed, 1 -> transposed

    long offset; // Where in X the matrix starts
    int lda;     // Stride between columns

    long X_size; // Length of X
    double* X;   // The matrix data
} matrix_tt;


matrix_tt* matrix_tt_init(const int m, const int n);
matrix_tt* matrix_tt_copy(const matrix_tt* A);
void matrix_tt_free(matrix_tt* A);
void matrix_tt_print(const matrix_tt* A, int just_the_matrix);

// Wrap a double* array as a matrix
matrix_tt* matrix_tt_wrap(int m, int n, double* X);

// Wrap a double* array with a pre-allocated matrix_tt
void matrix_tt_wrap_update(matrix_tt* A, int m, int n, double* X);

// Reshape the matrix A
void matrix_tt_reshape(int m, int n, matrix_tt* A);

// Copy data from matrix A to matrix B
void matrix_tt_copy_data(matrix_tt* B, const matrix_tt* A);

// Get a submatrix of a matrix, i.e. returns A[ii1:ii2-1, jj1:jj2-1] (does not allocate a new double* array)
matrix_tt* submatrix(const matrix_tt* A, int ii1, int ii2, int jj1, int jj2);

// Same as above, except allocates a new array
matrix_tt* submatrix_copy(const matrix_tt* A);

// Get a submatrix in-place
void submatrix_update(matrix_tt* A, int ii1, int ii2, int jj1, int jj2);

// Return the element A[ii,jj]
double matrix_tt_element(const matrix_tt* A, int ii, int jj);

/////////////////////////////////////////////// Linear algebra routines ///////////////////////////////////////////////

// Fill matrix with zero data
void matrix_tt_fill_zeros(matrix_tt* A);

// Get Frobenius norm
double frobenius_norm(matrix_tt* A);

// Multiply C = alpha*A*B + beta*C, where A and B are appropriately transposed
int matrix_tt_dgemm(const matrix_tt* A, const matrix_tt* B, matrix_tt* C, const double alpha, const double beta);

// Find least squares solution to Ax = b
void matrix_tt_dgels(matrix_tt* x, matrix_tt* A, const matrix_tt* b);

// In-place A=QR decomposition. Overwrites input matrix Q=A. If R=NULL, does not return diagonal part. Truncates to
// r columns.
int matrix_tt_truncated_qr(matrix_tt* Q, matrix_tt* R, int r);

//////////////////////////////////////////////////// MPI routines /////////////////////////////////////////////////////

// Send and receive a martrix
void matrix_tt_send(MPI_Comm comm, matrix_tt* buf, int dest);
void matrix_tt_recv(MPI_Comm comm, matrix_tt* buf, int source);

// Reduce a matrix A in-place. buf is used as storage for the operation, needing to be at least as large as A
void matrix_tt_reduce(MPI_Comm comm, int rank, matrix_tt* A, matrix_tt* buf, int head);

// Same as above, except only reduces among the ranks specified in group_ranks
void matrix_tt_group_reduce(MPI_Comm comm, int rank, matrix_tt* A, matrix_tt* buf, int head, int* group_ranks, int nranks);

// Performs allreduce in-place (not important for performance)
void matrix_tt_allreduce(MPI_Comm comm, matrix_tt* A);

// Broadcasts a matrix
void matrix_tt_broadcast(MPI_Comm comm, matrix_tt* buf);

// Fills a matrix with random Gaussian numbers
void matrix_tt_dlarnv(matrix_tt* A);

// Transposes the data of a matrix (whereas changing matrix->transpose does not move any data)
void row_to_col_major(const matrix_tt* mat_row, matrix_tt* mat_col);

#endif
