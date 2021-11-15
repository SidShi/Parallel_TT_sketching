

#ifndef SSTT_H
#define SSTT_H

#include "tt.h"
#include "tensor.h"
#include "VTime.h"
#include <gperftools/heap-profiler.h>

VTime* SSTT(tensor_train* tt, MPI_tensor* ten);

#endif