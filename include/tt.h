/* tt.h:
 /   This file contains everything related to the tensor_train struct
*/

#ifndef TENSOR_TRAIN_H
#define TENSOR_TRAIN_H

#include <lapacke.h>
#include <mpi.h>
#include "tensor.h"

typedef struct tensor_train {
    int d;           // dimension of tensor
    int* n;          // tensor size of each dimension
    int* r;          // tensor-train ranks

    double** trains; // array of pointers storing the address of the trains
} tensor_train;

tensor_train* TT_init(const int d, const int* restrict n);
tensor_train* TT_init_rank(const int d, const int* restrict n, const int* restrict r);
tensor_train* TT_copy(tensor_train* X);
void TT_free(tensor_train* sim);
void TT_print(tensor_train* tt);

// Broadcast a tensor train to all nodes
void tt_broadcast(MPI_Comm comm, tensor_train* tt);

// Find how much the tensor_train has compressed the tensor
double get_compression(const tensor_train* tt);

#endif
