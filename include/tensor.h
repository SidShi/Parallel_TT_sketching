/* tensor.h:
 /   This file contains everything related to constructing and working with the MPI_tensor data struct. This struct
 /   contains a subtensor of a full tensor.
*/

#include "matrix_tt.h"

#ifndef TENSOR_H
#define TENSOR_H

int* get_partition(int np, int n);

typedef struct MPI_tensor {
    int d;           // dimension of tensor
    int* n;          // tensor size of each dimension

    MPI_Comm comm;  // consists of all cores that are a part of the tensor
    int rank;       // The rank in the communicator
    int comm_size;  // The size in the communicator

    int** schedule;        // schedule[rank] is an n_schedule long list of the partitions that will be streamed
    int n_schedule;
    int* inverse_schedule; // an Nblocks long array which holds the information on who ``streams'' which subtensor - only used for static tensors

    // Partition[ii] is the partition of the tensor at the index ii.
    // Example:
    //     partitions[0] = [0,4,9]   (nps[0] = 2)
    //     partitions[1] = [0,3,6,9] (nps[1] = 3)
    //     partitions[2] = [0,9]     (nps[2] = 1)
    //   The subtensors, in column-major order, are
    //      X_sub0 = X[0:3, 0:2, 0:8]
    //      X_sub1 = X[4:8, 0:2, 0:8]
    //      X_sub2 = X[0:3, 3:5, 0:8]
    //      X_sub3 = X[4:8, 3:5, 0:8]
    //      X_sub4 = X[0:3, 6:8, 0:8]
    //      X_sub5 = X[4:8, 6:8, 0:8]
    int** partitions;
    int* nps;         // nps[ii] is the length of partition[ii]

    int current_part; // Which subtensor that is currently streamed.
                      // Set to -1 by default. 0 <= current_part < prod(nps)

    // The streaming function f_ten(X, ind1, ind2, parameters)
    // streams the subtensor X[ind1[0]:ind2[0]-1, ind1[1]:ind2[1]-1, ..., ind1[d-1]:ind2[d-1]-1]
    void (*f_ten)(double* restrict, int*, int*, const void*);
    void* parameters; // Parameters for the streaming function

    long X_size;    // Size of X at creation
    double* X;      // The subtensor owned by this core

    // Other temporary storage stuff
    int* ind1;
    int* ind2;
    int* tensor_part;
    int* group_ranks;
    int* t_kk;
} MPI_tensor;

MPI_tensor* MPI_tensor_init(int d, const int* n, int* nps, MPI_Comm comm,
                            void (*f_ten)(double* restrict, int*, int*, const void*), void* parameters);
MPI_tensor* MPI_tensor_copy(const MPI_tensor* ten);
void MPI_tensor_free(MPI_tensor* ten);


// Returns a column-major streaming schedule (used in the construction of MPI_tensors
int get_schedule(int** schedule, const int* nps, int d, int comm_size);

// Reshape an MPI_tensor with a new partition (used for adding a 'fictitious' dimension in SSTT)
void MPI_tensor_reshape(int d, const int* n, int n_partition, const long* partition, MPI_tensor* ten);

// Convert a tensor index [i[0], i[1], ..., i[d-1]] to vector index [i[0] + i[1]*n[0] + ... + i[d-1] * n[d-2] * ... * n[0]]
long to_vec_ind(const int* tensor_ind, const int* n, int d);
// The reverse of the above step
void to_tensor_ind(int* tensor_ind, long vec_ind, const int* n, int d);

// Stream a subtensor with block-index `t_v_block`.
void stream(MPI_tensor* ten, int t_v_block);

// Find which core will stream the block `t_v_block` and which streaming 'epoch' it will be streamed in. That is,
//   schedule[rank][epoch] = t_v_block
void MPI_tensor_get_owner(const MPI_tensor* ten, int t_v_block, int *rank, int *epoch);

// Get the subtensor entries
double* get_X(const MPI_tensor* ten);

// Print the tensor
void MPI_tensor_print(const MPI_tensor* ten, int flattening);

////////////////////////////////////////// Static (not streaming) tensor info //////////////////////////////////////////

// The subtensors of tensor loaded in memory, distributed across all cores.
// This is used in SSTT, so that we can still use all of the MPI_tensor sketching routines without having to define
// a new struct that isn't made for streaming. If f_ten* = NULL, the tensor assumes that it is static, and uses this
// struct
typedef struct p_static{
    double** subtensors; // The list of subtensors owned, should match the schedule
} p_static;

void* p_static_init(double** subtensors);
void p_static_free(void* parameters);

#endif