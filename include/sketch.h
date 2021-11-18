#ifndef SKETCH_H
#define SKETCH_H

#include "matrix_tt.h"
#include "tensor.h"

// A struct that has a bunch of tensor indexing info
// Notation for indices is
// x_y_description where
//     x in {t,f,s}, where t means full tensor info, f means flattening info, and s means "sketching" info (dual to the flattening info)
//     y in {v,t}, where v means vector-like indexing and t means tensor-like indexing
// Example: for an unfolding where we are interested in getting a column space, the flattening information pertains to
//   the 'i' index of the unfolding, and the sketching information pertains to the 'j' index. We have that
//   t_t_index = [f_t_index; s_t_index]
//   t_t_sizes = [f_t_sizes; s_t_sizes]
//   etc.
typedef struct flattening_info
{
    // Tensor info
    int t_d;        // Full tensor dimension (equal to ten->d)
    long t_N;       // Total number of entries of the subtensor (equal to the product of the sizes)
    int t_Nblocks;  // Total number of subtensor blocks
    int* t_nps;     // Length of partition in each direction (equal to ten->nps)

    int t_v_block;  // Block we have the info for (often equal to current_part)
    int* t_t_block;

    // Subtensor indices (i.e. subten = ten[t_t_index[0] + (0:t_t_sizes[0]-1), ..., t_t_index[d-1] + (0:t_t_sizes[d-1]-1)])
    int* t_t_index;
    int* t_t_sizes;

    // Flattening info
    int flattening; // The unfolding we are interested in
    int iscol;      // Indicator for whether we want a column or sketch
    int f_d;        // Number of dimensions we keep in the sketch
    long f_N;       // Length of unfolding in the dimension we keep
    int f_Nblocks;  // Number of subsketch blocks, equal to the product of f_nps
    int* f_nps;

    // Subsketch indices (i.e. subsketch = sketch[f_t_index[0] + (0:f_t_sizes[0]-1), ..., f_t_index[d-1] + (0:f_t_sizes[d-1]-1), 0:rank-1])
    int f_v_block;
    int* f_t_block;
    int* f_t_index;
    int* f_t_sizes;

    // Sketching dimensions info
    int s_d;    // Number of dimensions that are being sketched away
    long s_N;   // Length of the unfolding in the dimension we sketch away
    int* s_nps;

    // Dimension reduction map indices
    int* s_t_index;
    int* s_t_sizes;

} flattening_info;

flattening_info* flattening_info_init(const MPI_tensor* ten, int flattening, int iscol, int t_v_block);
void flattening_info_free(flattening_info* fi);
void flattening_info_print(flattening_info* fi);

// Update the flattening info for a new subtensor
void flattening_info_update(flattening_info* fi, const MPI_tensor* ten, int t_v_block);

// Update the flattening info for a new subsketch
void flattening_info_f_update(flattening_info* fi, const MPI_tensor* ten, int f_v_block);

// Object that can take a MPI_tensor object and sketch the column or row space of a given flattening.
typedef struct sketch {
    // Basic info
    MPI_tensor* ten;      // Contains all of the tensor information
    int flattening;       // Which flattening we take the sketch of
    int r;                // The rank of the sketch
    int buf;              // How many extra row/columns we use
    int iscol;            // Determines if the sketch is a column or row sketch
    int* owner_partition; // Determines which core owns which parts of the partition;

    // Big arrays
    long X_size;      // Size of X
    long lda;
    double* X;        // Array that stores the sketch, and subsequently the column/row space of the sketch.
    double* scratch;  // Size X_size array used for temporary work
    double* recv_buf; // Array used to receive when reducing

    // Dimension reduction tools
    matrix_tt** Omegas; // Arrays of random values used for the Khatri-Rao subspace iterations
    int d_KR;           // "middle" dimension of the Khatri-Rao product
    int stride_KR;      // Stride taken in the middle dimension.
    matrix_tt** KRs;    // (<1000 x r) arrays (see subtensor_khatri_rao for significance)

    // Indexing info
    flattening_info* fi;
} sketch;

sketch* sketch_init(MPI_tensor* ten, int flattening, int r, int buf, int iscol);
sketch* sketch_init_with_Omega(MPI_tensor* ten, int flattening, int r, int buf, int iscol, matrix_tt** Omegas);
void sketch_free(sketch* s);
sketch* sketch_copy(sketch* s);
void sketch_print(sketch* s);

// Given a block, find which core owns that block
int get_owner(int block, int* owner_partition, int comm_size);
int s_get_owner(sketch* s, int f_v_block);

// Get a subsketch that this core owns (otherwise, returns NULL)
matrix_tt* own_submatrix(sketch* s, int f_v_block, int with_buf);

// Multiply an MPI_tensor against the dimension reduction maps
void perform_sketch(sketch* s);
void multi_perform_sketch(sketch** sketches, int n_sketch);

// Find an orthogonal basis for the sketch
void sketch_qr(sketch* sketch);

// Communicate a sketch block
void sendrecv_sketch_block(matrix_tt* mat, sketch* s, flattening_info* fi, int recv_rank, int with_buf);

// Turn a sketch into an MPI_tensor object (used in SSTT)
// WARNING: This method clobbers the input sketch
MPI_tensor* sketch_to_tensor(sketch** s_ptr);

#endif
