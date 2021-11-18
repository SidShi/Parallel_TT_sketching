#ifndef PSTT_H
#define PSTT_H

#include "tensor.h"
#include "tt.h"
#include "matrix_tt.h"

// Find the tensor train decomposition `tt` of the tensor `ten` using the PSTT2 and PSTT2-onepass algorithms.
// If specified, takes the middle index to be mid. If mid=-1 (or any other invalid number), the program automatically
// chooses a good value for memory performance.
void PSTT2(tensor_train* tt, MPI_tensor* ten, int mid);
void PSTT2_onepass(tensor_train* tt, MPI_tensor* ten, int mid);

#endif
