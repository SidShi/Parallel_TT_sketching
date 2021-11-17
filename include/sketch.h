// Object that can take a MPI_tensor object and sketch the column or row space of a given flattening.

#ifndef SKETCH_H
#define SKETCH_H

#include "matrix.h"
#include "tensor.h"
#include "VTime.h"
#include <gperftools/heap-profiler.h>
// A struct that has a bunch of info that I need over and over again, and am tired of recomputing in every function
// Notation for indices is
// x_y_description where
//     x in {t,f,s}, where t means full tensor info, f means flattening info, and s means "sketching" info (dual to the flattening info)
//     y in {v,t}, where v means vector-like indexing and t means tensor-like indexing
typedef struct flattening_info
{
    // Tensor info
    int t_d;      // Full tensor dimension (equal to ten->d)
    long t_N;      // Total number of entries of the subtensor (equal to the product of the sizes)
    int t_Nblocks; // Total number of subtensor blocks
    int* t_nps;   // Length of partition in each direction (equal to ten->nps)

    int t_v_block;  // Block we have the info for (often equal to current_part)
    int* t_t_block;

    // Subtensor indices (i.e. subten = ten[t_t_index[0] + (0:t_t_sizes[0]-1), ..., t_t_index[d-1] + (0:t_t_sizes[d-1]-1)])
    int* t_t_index;
    int* t_t_sizes;

    // Flattening info
    int flattening;
    int iscol;
    int f_d;
    long f_N;
    int f_Nblocks;
    int* f_nps;

    int f_v_block;
    int* f_t_block;
    int* f_t_index;
    int* f_t_sizes;

    // Sketching dimensions info
    int s_d;
    long s_N;
    int* s_nps;

    int* s_t_index;
    int* s_t_sizes;

} flattening_info;

flattening_info* flattening_info_init(const MPI_tensor* ten, int flattening, int iscol, int t_v_block);
void flattening_info_free(flattening_info* fi);
void flattening_info_update(flattening_info* fi, const MPI_tensor* ten, int t_v_block);
void flattening_info_f_update(flattening_info* fi, const MPI_tensor* ten, int f_v_block);
void flattening_info_print(flattening_info* fi);

typedef struct sketch {
    MPI_tensor* ten;      // Contains all of the tensor information
    int flattening;       // Which flattening we take the sketch of
    int r;                // The rank of the sketch
    int buf;              // How many extra row/columns we use
    int iscol;            // Determines if the sketch is a column or row sketch
    int* owner_partition; // Determines which core owns which parts of the partition;


    long X_size; // Size of X
    long lda;
    double* X;   // Array that stores the sketch, and subsequently the column/row space of the sketch.
    double* scratch; // Size X_size array used for temporary work
    double* recv_buf; // Array used to receive when reducing

    matrix** Omegas; // Arrays of random values used for the Khatri-Rao subspace iterations

    int d_KR;        // "middle" dimension of the Khatri-Rao product
    int stride_KR;   // Stride taken in the middle dimension.
    matrix** KRs;    // (<1000 x r) arrays (see subtensor_khatri_rao for significance)

    flattening_info* fi;
} sketch;

sketch* sketch_init(MPI_tensor* ten, int flattening, int r, int buf, int iscol);
sketch* sketch_init_with_Omega(MPI_tensor* ten, int flattening, int r, int buf, int iscol, matrix** Omegas);
void sketch_free(sketch* s);
sketch* sketch_copy(sketch* s);
void sketch_print(sketch* s);

int get_owner(int block, int* owner_partition, int comm_size);
int s_get_owner(sketch* s, int f_v_block);
matrix* own_submatrix(sketch* s, int f_v_block, int with_buf);

void perform_sketch(sketch* s, VTime* tm);
void multi_perform_sketch(sketch** sketches, int n_sketch, VTime* tm);

void sketch_qr(sketch* sketch);

void sendrecv_sketch_block(matrix* mat, sketch* s, flattening_info* fi, int recv_rank, int with_buf);

MPI_tensor* sketch_to_tensor(sketch** s_ptr);

#endif
