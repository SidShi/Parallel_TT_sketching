/* tensor.h:
 /   This file contains everything related to constructing and working with the MPI_tensor data struct. This struct
 /   contains a part of a tensor. If the whole tensor X is indexed on [0, N), MPI_tensor contains the
 /   indices [N1, N2) within that tensor.
 /
 /   Note that the tensor struct of the openmp code corresponds to N1=0, N2=N for p==0, and N1=N2=N for p!=0. It will be
 /   assumed that if N1==N2, nothing really happens.
*/

#include "matrix.h"
#include <gperftools/heap-profiler.h>

#ifndef TENSOR_H
#define TENSOR_H

int* get_partition(int np, int n);

typedef struct MPI_tensor {
    int d;           // dimension of tensor
    int* n;          // tensor size of each dimension

    MPI_Comm comm;  // consists of all cores that are a part of the tensor
    int rank;       // The rank in the communicator
    int comm_size;       // The size in the communicator

    int** schedule;        // schedule[rank] is an n_schedule long list of the partitions that will be streamed
    int n_schedule;
    int* inverse_schedule; // an Nblocks long array which holds the information on who owns what - only used for static tensors

    int** partitions; // Partition[ii] is the partition of the tensor at the index ii
    int* nps;         // np[ii] is the length of partition[ii]

    int current_part; // What portion of the partition the tensor is currently filled with.
                      // Set to -1 by default. 0 <= current_part < np_1*np_2

    // The streaming function f_ten(X, ind1, ind2, parameters)
    // streams the subtensor X[ind1[0]:ind2[0]-1, ind1[1]:ind2[1]-1, ..., ind1[d-1]:ind2[d-1]-1]
    void (*f_ten)(double* restrict, int*, int*, const void*);
    void* parameters; // Parameters for the streaming function

    long X_size;     // Do not change! Size of X at creation
    double* X;      // the tensor owned by this processor, size partition[p+1] - partition[p];

    // Other temporary storage stuff
    int* ind1;
    int* ind2;
    int* tensor_part;
    int* group_ranks;
    int* t_kk;
} MPI_tensor;/**/

int get_schedule(int** schedule, const int* nps, int d, int comm_size);
MPI_tensor* MPI_tensor_init(int d, const int* n, int* nps, MPI_Comm comm,
                            void (*f_ten)(double* restrict, int*, int*, const void*), void* parameters);

MPI_tensor* MPI_tensor_copy(const MPI_tensor* ten);
void MPI_tensor_reshape(int d, const int* n, int n_partition, const long* partition, MPI_tensor* ten);
void MPI_tensor_free(MPI_tensor* ten);

typedef struct p_static{
    double** subtensors; // The list of subtensors
} p_static;

void* p_static_init(double** subtensors);
void p_static_free(void* parameters);


void to_tensor_ind(int* tensor_ind, long vec_ind, const int* n, int d);
long to_vec_ind(const int* tensor_ind, const int* n, int d);

void stream(MPI_tensor* ten, int part);
void MPI_tensor_get_owner(const MPI_tensor* ten, int t_v_block, int *rank, int *epoch);
double* get_X(const MPI_tensor* ten);

void MPI_tensor_print(const MPI_tensor* ten, int flattening);

long product(const int* n, int d);

long* col_partition(const int n_partition, const int d, const int* n, const int flattening);

long* flattening_sizes(MPI_tensor* ten, int index);

matrix* wrap_flattening(const MPI_tensor* ten, int index, int* ii_sub);
matrix* copy_flattening(MPI_tensor* ten, int index, int* ii_sub);

//MPI_tensor* MPI_tensor_gather(MPI_tensor* ten);

double MPI_tensor_error(MPI_tensor* ten_true, MPI_tensor* ten);

int* streaming_partition_assignment(MPI_Comm comm, int n_partition);

void flattening_truncated_qr(int flattening, const MPI_tensor* ten, matrix* Q, const int kr, const int BUF);

#endif