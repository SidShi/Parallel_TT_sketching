#ifndef SSTT_H
#define SSTT_H

#include "tt.h"
#include "tensor.h"

// Find the tensor train decomposition `tt` of the tensor `ten` using the SSTT algorithm.
// WARNING: ten is clobbered in this routine.
void SSTT(tensor_train* tt, MPI_tensor* ten);

#endif