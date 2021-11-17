// Functions used to take the tensor train in parallel!


#ifndef PSTT_H
#define PSTT_H

#include "tensor.h"
#include "tt.h"
#include "matrix.h"
#include "VTime.h"
#include <gperftools/heap-profiler.h>

VTime* PSTT2(tensor_train* tt, MPI_tensor* ten, int mid);
VTime* PSTT2_onepass(tensor_train* tt, MPI_tensor* ten, int mid);

#endif
