#include "matrix.h"
#include "tensor.h"
#include "sketch.h"
#include "VTime.h"

#include <stdio.h>
#include <stdlib.h>
#include <gperftools/heap-profiler.h>
#include <cblas.h>
#include <lapacke.h>

flattening_info* flattening_info_init(const MPI_tensor* ten, int flattening, int iscol, int t_v_block)
{
    // Tensor info
    flattening_info* fi = (flattening_info*) malloc(sizeof(flattening_info));
    int t_d = ten->d;

    int* t_nps = (int*) malloc(t_d*sizeof(int));
    int t_Nblocks = 1;
    for (int ii = 0; ii < t_d; ++ii){
        t_nps[ii] = ten->nps[ii];
        t_Nblocks = t_Nblocks * t_nps[ii];
    }

    int* t_t_block = (int*) malloc(t_d*sizeof(int));
    to_tensor_ind(t_t_block, t_v_block, t_nps, t_d);

    long t_N = 1;
    int* t_t_sizes = (int*) malloc(t_d*sizeof(int));
    int* t_t_index = (int*) malloc(t_d*sizeof(int));
    for (int ii = 0; ii < t_d; ++ii){
        int* partition_ii = ten->partitions[ii];
        t_t_index[ii] = partition_ii[t_t_block[ii]];
        t_t_sizes[ii] = partition_ii[t_t_block[ii]+1] - t_t_index[ii];
        t_N = t_N * t_t_sizes[ii];
    }

    // Flattening info
    int f_d = (iscol) ? flattening : t_d-flattening;

    int offset = (iscol) ? 0 : flattening;
    long f_N = 1;
    int f_Nblocks = 1;
    int* f_nps     = (int*) malloc(f_d * sizeof(int));
    int* f_t_block = (int*) malloc(f_d * sizeof(int));
    int* f_t_index = (int*) malloc(f_d * sizeof(int));
    int* f_t_sizes = (int*) malloc(f_d * sizeof(int));
    for (int ii = 0; ii < f_d; ++ii){
        f_nps[ii]     = t_nps[ii+offset];
        f_t_block[ii] = t_t_block[ii+offset];
        f_t_index[ii] = t_t_index[ii+offset];
        f_t_sizes[ii] = t_t_sizes[ii+offset];
        f_N = f_N * f_t_sizes[ii];
        f_Nblocks = f_Nblocks * f_nps[ii];
    }
    int f_v_block = to_vec_ind(f_t_block, f_nps, f_d);

    // Sketching dimensions info
    int s_d = t_d - f_d;
    long s_N = 1;

    offset = (iscol) ? flattening : 0;
    int* s_nps     = (int*) malloc(s_d * sizeof(int));
    int* s_t_index = (int*) malloc(s_d * sizeof(int));
    int* s_t_sizes = (int*) malloc(s_d * sizeof(int));
    for (int ii = 0; ii < s_d; ++ii){
        s_nps[ii]     = t_nps[ii+offset];
        s_t_index[ii] = t_t_index[ii+offset];
        s_t_sizes[ii] = t_t_sizes[ii+offset];
        s_N = s_N * s_t_sizes[ii];
    }

    // Assigning
    fi->t_d       = t_d;
    fi->t_N       = t_N;
    fi->t_Nblocks = t_Nblocks;
    fi->t_nps     = t_nps;
    fi->t_v_block = t_v_block;
    fi->t_t_block = t_t_block;
    fi->t_t_index = t_t_index;
    fi->t_t_sizes = t_t_sizes;

    fi->flattening = flattening;
    fi->iscol      = iscol;
    fi->f_d        = f_d;
    fi->f_N        = f_N;
    fi->f_Nblocks  = f_Nblocks;
    fi->f_nps      = f_nps;
    fi->f_v_block  = f_v_block;
    fi->f_t_block  = f_t_block;
    fi->f_t_index  = f_t_index;
    fi->f_t_sizes  = f_t_sizes;

    fi->s_d       = s_d;
    fi->s_N       = s_N;
    fi->s_nps     = s_nps;
    fi->s_t_index = s_t_index;
    fi->s_t_sizes = s_t_sizes;

    return fi;
}

void flattening_info_free(flattening_info* fi)
{
    free(fi->t_nps);     fi->t_nps = NULL;
    free(fi->t_t_block); fi->t_t_block = NULL;
    free(fi->t_t_index); fi->t_t_index = NULL;
    free(fi->t_t_sizes); fi->t_t_sizes = NULL;
    free(fi->f_nps);     fi->f_nps = NULL;
    free(fi->f_t_block); fi->f_t_block = NULL;
    free(fi->f_t_index); fi->f_t_index = NULL;
    free(fi->f_t_sizes); fi->f_t_sizes = NULL;
    free(fi->s_nps);     fi->s_nps = NULL;
    free(fi->s_t_index); fi->s_t_index = NULL;
    free(fi->s_t_sizes); fi->s_t_sizes = NULL;

    free(fi);
}

void flattening_info_update(flattening_info* fi, const MPI_tensor* ten, int t_v_block)
{
    // Tensor info
    int t_d = ten->d;
    int* t_nps = fi->t_nps;
    int* t_t_block = fi->t_t_block;
    to_tensor_ind(t_t_block, t_v_block, t_nps, t_d);

    long t_N = 1;
    int* t_t_sizes = fi->t_t_sizes;
    int* t_t_index = fi->t_t_index;
    for (int ii = 0; ii < t_d; ++ii){
        int* partition_ii = ten->partitions[ii];
        t_t_index[ii] = partition_ii[t_t_block[ii]];
        t_t_sizes[ii] = partition_ii[t_t_block[ii]+1] - t_t_index[ii];
        t_N = t_N * t_t_sizes[ii];
    }

    // Flattening info
    int flattening = fi->flattening;
    int iscol = fi->iscol;
    int f_d = fi->f_d;

    int offset = (iscol) ? 0 : flattening;
    long f_N = 1;
    int* f_nps     = fi->f_nps;
    int* f_t_block = fi->f_t_block;
    int* f_t_index = fi->f_t_index;
    int* f_t_sizes = fi->f_t_sizes;
    for (int ii = 0; ii < f_d; ++ii){
        f_nps[ii]     = t_nps[ii+offset];
        f_t_block[ii] = t_t_block[ii+offset];
        f_t_index[ii] = t_t_index[ii+offset];
        f_t_sizes[ii] = t_t_sizes[ii+offset];
        f_N = f_N * f_t_sizes[ii];
    }
    int f_v_block = to_vec_ind(f_t_block, f_nps, f_d);

    // Sketching dimensions info
    int s_d = fi->s_d;
    long s_N = 1;

    offset = (iscol) ? flattening : 0;
    int* s_nps     = fi->s_nps;
    int* s_t_index = fi->s_t_index;
    int* s_t_sizes = fi->s_t_sizes;
    for (int ii = 0; ii < s_d; ++ii){
        s_nps[ii]     = t_nps[ii+offset];
        s_t_index[ii] = t_t_index[ii+offset];
        s_t_sizes[ii] = t_t_sizes[ii+offset];
        s_N = s_N * s_t_sizes[ii];
    }

    // Assigning
    fi->t_d = t_d;
    fi->t_N = t_N;
    fi->t_nps = t_nps;
    fi->t_v_block = t_v_block;
    fi->t_t_block = t_t_block;
    fi->t_t_index = t_t_index;
    fi->t_t_sizes = t_t_sizes;

    fi->flattening = flattening;
    fi->iscol = iscol;
    fi->f_d = f_d;
    fi->f_N       = f_N;
    fi->f_nps     = f_nps;
    fi->f_v_block = f_v_block;
    fi->f_t_block = f_t_block;
    fi->f_t_index = f_t_index;
    fi->f_t_sizes = f_t_sizes;

    fi->s_d = s_d;
    fi->s_N = s_N;
    fi->s_nps = s_nps;
    fi->s_t_index = s_t_index;
    fi->s_t_sizes = s_t_sizes;
}

void flattening_info_f_update(flattening_info* fi, const MPI_tensor* ten, int f_v_block)
{
    // tensor info
    int t_d = fi->t_d;

    long t_N = -1;
    int t_v_block = -1;
    int* t_t_block = fi->t_t_block;
    int* t_t_index = fi->t_t_index;
    int* t_t_sizes = fi->t_t_sizes;
    for (int ii = 0; ii < t_d; ++ii){
        t_t_block[ii] = -1;
        t_t_index[ii] = 0;
        t_t_sizes[ii] = 0;
    }
    fi->t_N = t_N;
    fi->t_v_block = t_v_block;

    // flattening info
    int flattening = fi->flattening;
    int iscol = fi->iscol;
    int f_d = fi->f_d;
    int f_Nblocks = fi->f_Nblocks;
    int* f_nps = fi->f_nps;

    int* f_t_block = fi->f_t_block;
    to_tensor_ind(f_t_block, (long) f_v_block, f_nps, f_d);

    long f_N = 1;
    int* f_t_index = fi->f_t_index;
    int* f_t_sizes = fi->f_t_sizes;
    int offset = (iscol) ? 0 : flattening;
    for (int ii = 0; ii < f_d; ++ii){
        int* partition_ii = ten->partitions[ii + offset];
        f_t_index[ii] = partition_ii[f_t_block[ii]];
        f_t_sizes[ii] = partition_ii[f_t_block[ii]+1] - f_t_index[ii];
        f_N = f_N * f_t_sizes[ii];
    }
    fi->f_N = f_N;
    fi->f_v_block = f_v_block;

    // Sketching dimensions info
    int s_d = fi->s_d;
    long s_N = 0;

    int* s_t_index = fi->s_t_index;
    int* s_t_sizes = fi->s_t_sizes;

    for (int ii = 0; ii < s_d; ++ii){
        s_t_index[ii] = 0;
        s_t_sizes[ii] = 0;
    }
    fi->s_N = s_N;
}

void flattening_info_print(flattening_info* fi)
{
    printf("\n~~~~~~~~~~~~~~~ Flattening Info ~~~~~~~~~~~~~~~\n");
    int t_d = fi->t_d;
    printf("t_d = %d\n", t_d);

    printf("t_N = %ld\n", fi->t_N);

    printf("t_Nblocks = %d\n", fi->t_Nblocks);

    if (t_d>0){
        printf("t_nps = [%d", fi->t_nps[0]);
        for (int ii = 1; ii < t_d; ++ii){printf(", %d", fi->t_nps[ii]); }
        printf("]\n");
    }
    else{ printf("t_nps = []\n"); }

    printf("t_v_block = %d\n", fi->t_v_block);

    if (t_d>0){
        printf("t_t_block = [%d", fi->t_t_block[0]);
        for (int ii = 1; ii < t_d; ++ii){printf(", %d", fi->t_t_block[ii]);}
        printf("]\n");
    }
    else{ printf("t_t_block = []\n"); }

    if (t_d>0){
        printf("t_t_index = [%d", fi->t_t_index[0]);
        for (int ii = 1; ii < t_d; ++ii){printf(", %d", fi->t_t_index[ii]);}
        printf("]\n");
    }
    else{ printf("t_t_index = []\n"); }

    if (t_d>0){
        printf("t_t_sizes = [%d", fi->t_t_sizes[0]);
        for (int ii = 1; ii < t_d; ++ii){printf(", %d", fi->t_t_sizes[ii]);}
        printf("]\n");
    }
    else{ printf("t_t_sizes = []\n"); }

    printf("flattening = %d\n", fi->flattening);

    printf("iscol = %d\n", fi->iscol);

    int f_d = fi->f_d;
    printf("f_d = %d\n", f_d);

    printf("f_N = %ld\n", fi->f_N);

    printf("f_Nblocks = %d\n", fi->f_Nblocks);

    if (f_d>0){
        printf("f_nps = [%d", fi->f_nps[0]);
        for (int ii = 1; ii < f_d; ++ii){printf(", %d", fi->f_nps[ii]);}
        printf("]\n");
    }
    else{ printf("f_nps = []\n"); }

    printf("f_v_block = %d\n", fi->f_v_block);

    if (f_d>0){
        printf("f_t_block = [%d", fi->f_t_block[0]);
        for (int ii = 1; ii < f_d; ++ii){printf(", %d", fi->f_t_block[ii]);}
        printf("]\n");
    }
    else{ printf("f_t_block = []\n"); }

    if (f_d>0){
        printf("f_t_index = [%d", fi->f_t_index[0]);
        for (int ii = 1; ii < f_d; ++ii){printf(", %d", fi->f_t_index[ii]);}
        printf("]\n");
    }
    else{ printf("f_t_index = []\n"); }

    if (f_d>0){
        printf("f_t_sizes = [%d", fi->f_t_sizes[0]);
        for (int ii = 1; ii < f_d; ++ii){printf(", %d", fi->f_t_sizes[ii]);}
        printf("]\n");
    }
    else{ printf("f_t_sizes = []\n"); }

    int s_d = fi->s_d;
    printf("s_d = %d\n", s_d);

    printf("s_N = %ld\n", fi->s_N);

    if (s_d>0){
        printf("s_nps = [%d", fi->s_nps[0]);
        for (int ii = 1; ii < s_d; ++ii){printf(", %d", fi->s_nps[ii]);}
        printf("]\n");
    }
    else{ printf("s_nps = []\n"); }

    if (s_d>0){
        printf("s_t_index = [%d", fi->s_t_index[0]);
        for (int ii = 1; ii < s_d; ++ii){printf(", %d", fi->s_t_index[ii]);}
        printf("]\n");
    }
    else{ printf("s_t_index = []\n"); }


    if (s_d>0){
        printf("s_t_sizes = [%d", fi->s_t_sizes[0]);
        for (int ii = 1; ii < s_d; ++ii){printf(", %d", fi->s_t_sizes[ii]);}
        printf("]\n");
    }
    else{ printf("s_t_sizes = []\n"); }
    printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");
}


int* get_sketch_owners(MPI_tensor* ten, int flattening, int iscol)
{
    int np_flattening = 1;
    int* nps = ten->nps;
    int ii0 = (iscol) ? 0 : flattening;
    int ii1 = (iscol) ? flattening : ten->d;

    for (int ii = ii0; ii < ii1; ++ii){
        np_flattening = np_flattening * nps[ii];
    }

    int* owners = get_partition(ten->comm_size, np_flattening);
    return owners;
}

void get_sketch_height(MPI_tensor* ten, int* owners, int flattening, int iscol, long* X_height_ptr, long* buf_height_ptr)
{
    *X_height_ptr = 0;
    *buf_height_ptr = 0;

    flattening_info* fi = flattening_info_init(ten, flattening, iscol, 0);

    for (int rank = 0; rank < ten->comm_size; ++rank){
        long X_height_tmp = 0;
        for (int jj = owners[rank]; jj < owners[rank + 1]; ++jj){
            flattening_info_f_update(fi, ten, jj);
            X_height_tmp = X_height_tmp + fi->f_N;
            *buf_height_ptr = (fi->f_N > *buf_height_ptr) ? fi->f_N : *buf_height_ptr;
        }

        *X_height_ptr = (X_height_tmp > *X_height_ptr) ? X_height_tmp : *X_height_ptr;
    }

    flattening_info_free(fi);
}

matrix** get_sketch_Omega(const MPI_tensor* ten, int flattening, int r, int buf, int iscol)
{
    int ii0 = iscol ? flattening : 0;
    int n_Omega = iscol ? (ten->d) - flattening : flattening;

    matrix** Omegas = (matrix**) malloc(n_Omega * sizeof(matrix*));
    for (int ii = 0; ii < n_Omega; ++ii){
        Omegas[ii] = matrix_init(ten->n[ii+ii0], r+buf);
        matrix_dlarnv(Omegas[ii]);
    }

    return Omegas;
}

void get_KR_info(int *d_KR, int *stride_KR, matrix** KRs, MPI_tensor* ten, int flattening, int r, int buf, int iscol)
{
    int max_m = 1000;


    int assigned = 0;
    *d_KR = 0;
    int N_KR = 1;

    int ii0 = iscol ? flattening : 0;
    int n_Omega = iscol ? (ten->d) - flattening : flattening;

    KRs[2] = matrix_init(1, r+buf);

    for (int ii = 0; ii < n_Omega; ++ii){
        if (assigned == 0){
            int* partition_ii = ten->partitions[ii + ii0];
            int sz_ii = 0;
            for (int jj = 0; jj < ten->nps[ii+ii0]; ++jj){
                int candidate_n = partition_ii[jj+1] - partition_ii[jj];
                sz_ii = (sz_ii > candidate_n) ? sz_ii : candidate_n;
            }
            if (N_KR * sz_ii < max_m){
                *d_KR = ii+1;
                N_KR = N_KR * sz_ii;
            }
            else{
                *stride_KR = max_m / N_KR;
                KRs[0] = matrix_init(N_KR, r+buf);
                KRs[1] = matrix_init(*stride_KR, r+buf);
                N_KR = N_KR * (*stride_KR);
//                printf("N_KR = %d, stride_KR = %d\n", N_KR, *stride_KR);
                KRs[3] = matrix_init(N_KR, r+buf);
//                printf("Done allocating\n");
                assigned = 1;
            }
        }
    }
    if (assigned == 0){
        *stride_KR = 0;
        KRs[0] = matrix_init(N_KR, r+buf);
        KRs[1] = matrix_init(1, r+buf);
        KRs[3] = matrix_init(N_KR, r+buf);
    }
}


sketch* sketch_init(MPI_tensor* ten, int flattening, int r, int buf, int iscol)
{
    sketch* s = (sketch*) malloc(sizeof(sketch));

    s->ten = ten;
    s->flattening = flattening;
    s->r = r;
    s->buf = buf;
    s->iscol = iscol;
    s->owner_partition = get_sketch_owners(ten, flattening, iscol);
    long recv_buf_height = 0;
    get_sketch_height(ten, s->owner_partition, flattening, iscol, &(s->lda), &recv_buf_height);
    s->Omegas = get_sketch_Omega(ten, flattening, r, buf, iscol);
    s->X_size = (s->lda)*(r+buf);
    s->X = (double*) calloc(s->X_size, sizeof(double));
    s->scratch = (double*) calloc(s->X_size, sizeof(double));
    s->recv_buf = (double*) calloc(recv_buf_height * (r+buf), sizeof(double));
    s->fi = flattening_info_init(ten, flattening, iscol, 0);
    s->KRs = (matrix**) calloc(4, sizeof(matrix*));
    get_KR_info(&(s->d_KR), &(s->stride_KR), s->KRs, ten, flattening, r, buf, iscol);

//    printf("s->X_size = %ld, recv_buf size = %ld\n", s->X_size, recv_buf_height * (r+buf));

    return s;
}

//(const MPI_tensor* ten, int flattening, int r, int buf, int iscol)
matrix** copy_sketch_Omega(matrix** Omegas, MPI_tensor* ten, int flattening, int r, int buf, int iscol)
{
    int ii0 = iscol ? flattening : 0;
    int n_Omega = iscol ? (ten->d) - flattening : flattening;

    matrix** Omegas_cp = (matrix**) malloc(n_Omega * sizeof(matrix*));
    for (int ii = 0; ii < n_Omega; ++ii){
        submatrix_update(Omegas[ii], 0, ten->n[ii+ii0], 0, r+buf);
        Omegas_cp[ii] = matrix_copy(Omegas[ii]);
    }

    return Omegas_cp;
}


sketch* sketch_init_with_Omega(MPI_tensor* ten, int flattening, int r, int buf, int iscol, matrix** Omegas)
{
    sketch* s = (sketch*) malloc(sizeof(sketch));
    s->ten = ten;
    s->flattening = flattening;
    s->r = r;
    s->buf = buf;
    s->iscol = iscol;
    s->owner_partition = get_sketch_owners(ten, flattening, iscol);
    long recv_buf_height;
    get_sketch_height(ten, s->owner_partition, flattening, iscol, &(s->lda), &recv_buf_height);
    s->Omegas = copy_sketch_Omega(Omegas, ten, flattening, r, buf, iscol);
    s->X_size = (s->lda)*(r+buf);
    s->X = (double*) calloc(s->X_size, sizeof(double));
    s->scratch = (double*) calloc(s->X_size, sizeof(double));
    s->recv_buf = (double*) calloc(recv_buf_height * (r+buf), sizeof(double));
    s->fi = flattening_info_init(ten, flattening, iscol, 0);
    s->KRs = (matrix**) calloc(4, sizeof(matrix*));
    get_KR_info(&(s->d_KR), &(s->stride_KR), s->KRs, ten, flattening, r, buf, iscol);

    return s;
}


// NOTE: does not free the MPI_tensor. We assume this is shared, and should not be freed this way.
void sketch_free(sketch* s)
{
    MPI_tensor* ten = s->ten; s->ten = NULL;
    int d = ten->d;
    int n_Omega = s->iscol ? d - s->flattening : s->flattening;
    for (int ii = 0; ii < n_Omega; ++ii){
        matrix_free(s->Omegas[ii]); s->Omegas[ii] = NULL;
    }
    for (int ii = 0; ii < 4; ++ii){
        matrix_free(s->KRs[ii]); s->KRs[ii] = NULL;
    }
    free(s->KRs); s->KRs = NULL;

    free(s->owner_partition); s->owner_partition = NULL;
    free(s->X);               s->X = NULL;
    free(s->scratch);         s->scratch = NULL;
    free(s->Omegas);          s->Omegas = NULL;
    free(s->recv_buf);        s->recv_buf = NULL;
    flattening_info_free(s->fi); s->fi = NULL;
    free(s);
}

//sketch* sketch_copy(sketch* s)
//{
//    sketch* s_copy = (sketch*) malloc(sizeof(sketch));
//    s_copy->ten = s->ten;
//    s_copy->flattening = s->flattening;
//}
//
void sketch_print(sketch* s)
{
    printf("Printing sketch at %p\n", s);

    printf("\nten address: %p\n", s->ten);
    printf("flattening = %d\n",s->flattening);
    printf("r = %d\n", s->r);
    printf("buf = %d\n", s->buf);
    printf("iscol = %d\n", s->iscol);

    int* op = s->owner_partition;
    MPI_tensor* ten = s->ten;
    printf("owner_partition = [%d", op[0]);
    for (int ii = 1; ii < (ten->comm_size) + 1; ++ii){
        printf(", %d", op[ii]);
    }
    printf("]\n");

    printf("lda = %ld\n", s->lda);
    printf("X_size = %ld\n", s->X_size);

    printf("\nX = \n");
    int r = s->r; int buf = s->buf; long X_size = s->X_size; long lda = s->lda;
    matrix* X_mat = matrix_wrap(lda, r+buf, s->X);
    matrix_print(X_mat, 1);
    free(X_mat); X_mat = NULL;

    printf("\nscratch = \n");
    matrix* scratch_mat = matrix_wrap(lda, r+buf, s->scratch);
    matrix_print(scratch_mat, 1);
    free(scratch_mat); scratch_mat = NULL;

    int n_Omega = s->iscol ? (s->ten)->d-s->flattening : s->flattening;
    for (int ii = 0; ii < n_Omega; ++ii){
        printf("\nOmegas[%d] = \n", ii);
        matrix_print(s->Omegas[ii], 1);
    }
}

// Performs C[ii + n_ii * jj,kk] = A[ii, kk] * B[jj, kk]
void submatrix_khatri_rao_outer_product(matrix* A, matrix* B, matrix* C)
{
    for (int kk = 0; kk < A->n; ++kk){
        int A_offset = A->offset + kk*(A->lda);
        int B_offset = B->offset + kk*(B->lda);
        int C_offset = C->offset + kk*(C->lda);


        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                A->m, B->m, 1,
                1.0,
                A->X + A_offset, A->m,
                B->X + B_offset, 1,
                0.0,
                C->X + C_offset, A->m);

//        for (int jj = 0; jj < B->m; ++jj){
//            double B_jj = matrix_element(B, jj, kk);
//            for (int ii = 0; ii < A->m; ++ii){
//                C->X[C_offset + jj*(A->m) + ii] = B_jj * A->X[A_offset + ii];
//            }
//        }
    }
}

// General idea of the algorithm:
// Split the Omegas into 3 parts: ii < d_KR, ii == d_KR, ii > d_KR
// For ii <  d_KR, we will pre-multiply the sketch matrices Omega[ii] (Call this KR_1)
// For ii == d_KR, we will loop through blocks of Omega[d_KR]
// For ii > d_KR, we will loop and multiply to get a row vector       (Call this KR_3)

// Then, at each point in the loop, we will perform
//     KR_2 = Omegas[ii] * KR_3 and
//     KR_4 = KR_1 * KR_2,
// where * is the Khatri-Rao product. Finally, we multiply X_mat by KR_4 to update the sketch, and repeat.
void subtensor_khatri_rao(sketch* s, matrix* C, flattening_info* fi, double beta, matrix* X_mat)
{
    int r = s->r + s->buf;
    matrix** Omegas = s->Omegas;

    MPI_tensor* ten = s->ten;
    int d_KR = s->d_KR;
    matrix* KR_1;
    matrix* KR_4;
    if ((d_KR == 0) || (d_KR%2 == 1)){
        KR_1 = s->KRs[0];
        KR_4 = s->KRs[3];
    }
    else{
        KR_1 = s->KRs[3];
        KR_4 = s->KRs[0];
    }
    int s_d = fi->s_d;

    int N_kk = 1;

    int h = 0;

    int Omega_offset_KR;

//    printf("Starting loop to get KR_1\n");
    for (int ii = 0; ii < s_d; ++ii){
        int ii0 = fi->s_t_index[ii];
        int ii1 = ii0 + fi->s_t_sizes[ii];

        submatrix_update(Omegas[ii], ii0, ii1, 0, r);

        if (ii > d_KR){
            N_kk = N_kk * fi->s_t_sizes[ii]; // Number of elements in outer loop
        }
        else if (ii == d_KR){
            Omega_offset_KR = ii0;
        }


        if ((ii == 0) && (d_KR > 0)){
            matrix_reshape(ii1-ii0, r, KR_1);
            matrix_copy_data(KR_1, Omegas[0]);
            h = ii1 - ii0;
        }
        else if (ii < d_KR){ // Multiply inner matrices
            matrix* tmp = KR_1;
            KR_1 = KR_4;
            KR_4 = tmp;

            h = h*(ii1-ii0);
            matrix_reshape(h, r, KR_1);
            submatrix_khatri_rao_outer_product(KR_4, Omegas[ii], KR_1);
        }
    }
//    printf("Finished loop to get KR_1\n");

    if (fi->iscol){
        matrix_wrap_update(X_mat, fi->f_N, fi->s_N, get_X(ten));
    }
    else{
        matrix_wrap_update(X_mat, fi->s_N, fi->f_N, get_X(ten));
    }
    X_mat->transpose = (fi->iscol ? 0 : 1);

    if (d_KR == s_d){ // If we have done everything, just multiply (the sketch is relatively small)
//        printf("\n~~~~~~~~~\nMultiplying\n~~~~~~~~~~~\nKR_1 = \n");
//        matrix_print(X_mat, 1);
//        printf("\nKR_1 = \n");
//        matrix_print(KR_1, 1);
        matrix_dgemm(X_mat, KR_1, C, 1.0, beta);
        return;
    }

    int stride_KR = s->stride_KR;
    matrix* KR_2 = s->KRs[1];
    matrix* KR_3 = s->KRs[2];

    int* t_kk = ten->t_kk;
    int size_KR = fi->s_t_sizes[d_KR];
    int N_ll = 1 + (size_KR - 1) / stride_KR;

    int tensor_offset = 0;

//    printf("Starting multiplication loop\n");
    for (int kk = 0; kk < N_kk; ++kk) {
        to_tensor_ind(t_kk, (long) kk, fi->s_t_sizes + d_KR + 1, s_d - 1 - d_KR);

//        printf("kk%d got t_kk\n", kk);
        for (int ii = d_KR + 1; ii < s_d; ++ii){
            if (ii == d_KR + 1){
                for (int jj = 0; jj < r; ++jj){
                    KR_3->X[jj] = matrix_element(Omegas[ii], t_kk[ii-d_KR-1], jj);
                }
            }
            else{
                for (int jj = 0; jj < r; ++jj){
                    KR_3->X[jj] *= matrix_element(Omegas[ii], t_kk[ii-d_KR-1], jj);
                }
            }
        }

//        printf("kk%d got KR_3\n", kk);
        for (int ll = 0; ll < N_ll; ++ll){
            double bb = ((kk==0) && (ll==0)) ? beta : 1.0;
            int ii0 = Omega_offset_KR + ll * stride_KR;
            int ii1 = ((ll+1)*stride_KR > size_KR) ? size_KR + Omega_offset_KR: (ll+1)*stride_KR + Omega_offset_KR;

            submatrix_update(Omegas[d_KR], ii0, ii1, 0, r);
            if (d_KR + 1 == s_d){ // If we only need KR_1 and Omegas[ii]
//                printf("d_KR + 1 == s_d\n");
                matrix_reshape((KR_1->m) * (ii1-ii0), r, KR_4);
                submatrix_khatri_rao_outer_product(KR_1, Omegas[d_KR], KR_4);
            }
            else if (d_KR == 0){ // If we only need Omegas[ii] and KR_3
//                printf("d_KR == 0, Reshaping KR_4\n");
                matrix_reshape(Omegas[d_KR]->m, r, KR_4);
                submatrix_khatri_rao_outer_product(Omegas[d_KR], KR_3, KR_4);
            }
            else{ // We need everything...
//                printf("Reshaping KR_2\n");
                matrix_reshape(ii1-ii0, r, KR_2);
                submatrix_khatri_rao_outer_product(Omegas[d_KR], KR_3, KR_2);
//                printf("Reshaping KR_4\n");
                matrix_reshape((ii1-ii0) * (KR_1->m), r, KR_4);
                submatrix_khatri_rao_outer_product(KR_1, KR_2, KR_4);
            }


            if (fi->iscol) {
                submatrix_update(X_mat, 0, fi->f_N, tensor_offset, tensor_offset + KR_4->m);
            }
            else {
                submatrix_update(X_mat, tensor_offset, tensor_offset + KR_4->m, 0, fi->f_N);
            }

//            printf("\nPerforming dgemm in submatrix_khatri_rao\n");
//            printf("KR_4 = \n");
//            matrix_print(KR_4, 1);
//            printf("X_mat = \n");
//            matrix_print(X_mat, 1);
            matrix_dgemm(X_mat, KR_4, C, 1.0, bb);
            tensor_offset = tensor_offset + KR_4->m;
        }
    }
}

int get_owner(int block, int* owner_partition, int comm_size)
{
    int color = -1;
    for (int ii = 0; ii < comm_size; ++ii){
        if ((block >= owner_partition[ii]) && (block < owner_partition[ii+1])){
            color = ii;
        }
    }

    return color;
}

int s_get_owner(sketch* s, int f_v_block){
    int* op = s->owner_partition;
    MPI_tensor* ten = s->ten;
    int size = ten->comm_size;

    // Binary search (basically wikipedia)
    int a = 0;
    int b = size-1;
    int c;

    while (a <= b){
        c = (a+b)/2;
        if (f_v_block < op[c]){
            b = c-1;
        }
        else if(f_v_block >= op[c+1]){
            a = c+1;
        }
        else{
            return c;
        }
    }

    return -1;
}

void own_submatrix_update(matrix* mat, sketch* s, int f_v_block, int with_buf){
    if (f_v_block == -1){
        return;
    }

    int sketch_offset = 0;
    int sketch_lda = s->lda;
    MPI_tensor* ten = s->ten;
    int world_rank = ten->rank;
    int* owner_partition = s->owner_partition;

    flattening_info* fi = s->fi;
    if ((f_v_block < owner_partition[world_rank]) || (f_v_block >= owner_partition[world_rank+1])){
        printf("r%d own_submatrix: This core does not own f_v_block = %d\n", world_rank, f_v_block);
    }

    for (int block = s->owner_partition[world_rank]; block < f_v_block; ++block){
        flattening_info_f_update(fi, ten, block);
        sketch_offset = sketch_offset + fi->f_N;
    }
    flattening_info_f_update(fi, ten, f_v_block);


    matrix_wrap_update(mat, sketch_lda, s->r + s->buf, s->X);
    int r = (with_buf) ? s->r + s->buf : s->r;
    submatrix_update(mat, sketch_offset, sketch_offset + fi->f_N, 0, r);
}

matrix* own_submatrix(sketch* s, int f_v_block, int with_buf){
    matrix* mat = (matrix*) malloc(sizeof(matrix));
    own_submatrix_update(mat, s, f_v_block, with_buf);
    return mat;
}

void subtensor_sketch_multiply(sketch* s, flattening_info* fi, matrix** holders)
{
    MPI_tensor* ten = s->ten;
    int* owner_partition = s->owner_partition;
    int world_size = ten->comm_size;
    int world_rank = ten->rank;

    // Actually sketch the thing
    matrix* sketch_mat = holders[0];
    double beta = 0.0;
    if (ten->current_part != -1){
        int current_color = get_owner(fi->f_v_block, owner_partition, world_size);
        if (current_color == world_rank){
            own_submatrix_update(sketch_mat, s, fi->f_v_block, 1);
            beta = 1.0;
        }
        else{
            matrix_wrap_update(sketch_mat, fi->f_N, s->r + s->buf, s->scratch);
            beta = 0.0;
        }

        subtensor_khatri_rao(s, sketch_mat, fi, beta, holders[1]);
    }
}

void subtensor_sketch_communicate(sketch* s, int stream_step, flattening_info* fi, matrix** holders)
{
    MPI_tensor* ten = s->ten;
    int* owner_partition = s->owner_partition;
    int world_size = ten->comm_size;
    int world_rank = ten->rank;
    int* group_ranks = ten->group_ranks;

    flattening_info* fi_tmp = s->fi;
    matrix* sketch_mat = holders[0];

    // For each flattening block
    for (int ii = 0; ii < fi->f_Nblocks; ++ii){
        int kk = 0;
        int owner_ii = get_owner(ii, owner_partition, world_size);
        int group_owner;
        // For each rank in the world
        for (int jj = 0; jj < world_size; ++jj){
            // If the rank owns the block
            if (owner_ii == jj){
                // Add it to the group
                group_ranks[kk] = jj;
                kk = kk+1;
                group_owner = jj;
            }
            else{
                int* schedule_jj = ten->schedule[jj];
                int stream_jj = schedule_jj[stream_step];
                if (stream_jj != -1){
                    flattening_info_update(fi_tmp, ten, schedule_jj[stream_step]);
                    // Else, if the rank just calculated something that adds to the block
                    if (ii == fi_tmp->f_v_block){
                        // Add it to the group
                        group_ranks[kk] = jj;
                        kk = kk+1;
                    }
                }
            }
        }


        if (world_rank == owner_ii){
            matrix* comm_matrix = holders[1];
            own_submatrix_update(comm_matrix, s, ii, 1);
            matrix* buf_mat = holders[2];
            matrix_wrap_update(buf_mat, comm_matrix->m, comm_matrix->n, s->recv_buf);

            matrix_group_reduce(ten->comm, world_rank, comm_matrix, buf_mat, owner_ii, group_ranks, kk);
        }
        else if (sketch_mat->X){
            matrix* buf_mat = holders[2];
            matrix_wrap_update(buf_mat, sketch_mat->m, sketch_mat->n, s->recv_buf);
            matrix_group_reduce(ten->comm, world_rank,sketch_mat, buf_mat, owner_ii, group_ranks, kk);
        }

    }
}


void multi_perform_sketch(sketch** sketches, int n_sketch, VTime* tm)
{
// matrix* sketch_mat = NULL;
// matrix* comm_matrix = own_submatrix(s, ii, 1);
// matrix* buf_mat = matrix_wrap(comm_matrix->m, 1, s->recv_buf);
    int n_holders = 3;
    matrix** holders = (matrix**) malloc(n_holders*n_sketch*sizeof(matrix*));
    for (int ii = 0; ii < n_holders*n_sketch; ++ii){
        holders[ii] = (matrix*) calloc(1, sizeof(matrix));
    }

    double t_break = 0;
    if (tm != NULL){
         t_break = tm->t_break;
    }
    VTime_break(tm, -1, NULL);

    MPI_tensor* ten = sketches[0]->ten; // Tensor
    int rank = ten->rank;
    int* schedule_rank = ten->schedule[rank];

    flattening_info** fis = (flattening_info**) calloc(n_sketch, sizeof(flattening_info*));
    for (int ii = 0; ii < n_sketch; ++ii){
        fis[ii] = flattening_info_init(ten, sketches[ii]->flattening, sketches[ii]->iscol, 0);
    }



    for (int ii = 0; ii < ten->n_schedule; ++ii){
        stream(ten, schedule_rank[ii]);
//        MPI_tensor_print(ten, 1);
        VTime_break(tm, 0, "multi_perform_sketch - time spent streaming");
        for (int jj = 0; jj < n_sketch; ++jj){
            matrix** holders_jj = holders + jj*n_holders;
            flattening_info_update(fis[jj], ten, ten->current_part);
            holders_jj[0]->X = NULL;
            subtensor_sketch_multiply(sketches[jj], fis[jj], holders_jj);
        }
        VTime_break(tm, 1, "multi_perform_sketch - time spent multiplying");
        for (int jj = 0; jj < n_sketch; ++jj){
            matrix** holders_jj = holders + jj*n_holders;
            subtensor_sketch_communicate(sketches[jj], ii, fis[jj], holders_jj);
        }
        VTime_break(tm, 2, "multi_perform_sketch - time spent communicating");
    }

    if (tm != NULL){
        tm->t_break = t_break;
    }

    for (int ii = 0; ii < n_holders*n_sketch; ++ii){
        free(holders[ii]); holders[ii] = NULL;
    }

    for (int ii = 0; ii < n_sketch; ++ii){
        flattening_info_free(fis[ii]); fis[ii] = NULL;
    }
    free(fis); fis = NULL;
    free(holders); holders = NULL;
}

void sketch_qr(sketch* sketch)
{

    MPI_tensor* ten = sketch->ten;
    int* owner_partition = sketch->owner_partition;
    int flattening = sketch->flattening;
    int iscol = sketch->iscol;
    int r = sketch->r;
    int buf = sketch->buf;

    MPI_Comm comm = ten->comm;
    int size = ten->comm_size;
    int rank = ten->rank;

    // A little pre-processing
    int head = 0;
    flattening_info* fi = flattening_info_init(ten, flattening, iscol, 0);
    int* Ns = (int*) calloc(size, sizeof(int));

    for (int ii = 0; ii < size; ++ii){
        for (int jj = owner_partition[ii]; jj < owner_partition[ii+1]; ++jj){
            flattening_info_f_update(fi, ten, jj);
            Ns[ii] = Ns[ii] + fi->f_N;
        }
    }

    int N_rank = Ns[rank];
    int lda = sketch->lda;
    matrix* Q_big = matrix_wrap(lda, r+buf, sketch->X);
    matrix* Q = submatrix(Q_big, 0, N_rank, 0, r+buf);
    matrix* Q_head = NULL;
    matrix* R = NULL;

    if (rank == head){
        Q_head = matrix_init(size * (r+buf), r+buf);
        R = submatrix(Q_head, 0, r+buf, 0, r+buf);
    }
    else{
        R = matrix_init(r+buf, r+buf);
    }

    matrix_truncated_qr(Q, R, r+buf);



    // Gather the Rs
    for (int jj = 0; jj < r+buf; ++jj){
//        printf("r%d jj%d\n", rank, jj);
        if (rank == head){
            double* col = Q_head->X + jj*(Q_head->lda);
            MPI_Gather(col, r+buf, MPI_DOUBLE, col, r+buf, MPI_DOUBLE, head, comm);
        }
        else{
            MPI_Gather(R->X + jj*(R->lda), r+buf, MPI_DOUBLE, NULL, r+buf, MPI_DOUBLE, head, comm);
        }
    }

    // Take the QR of the Rs
    if (rank == head){
        matrix_truncated_qr(Q_head, NULL, r+buf);
    }

    // Scatter the Q of the preceding step
    for (int jj = 0; jj < r+buf; ++jj){
        if (rank == head){
            double* col = Q_head->X + jj*(Q_head->lda);
            MPI_Scatter(col, r+buf, MPI_DOUBLE, col, r+buf, MPI_DOUBLE, head, comm);
        }
        else{
            MPI_Scatter(NULL, r+buf, MPI_DOUBLE, R->X + jj*(R->lda), r+buf, MPI_DOUBLE, head, comm);
        }
    }

    // Multiply to get the final Q
    matrix* X_big = matrix_wrap(lda, r, sketch->scratch);
    matrix* new_X = submatrix(X_big, 0, N_rank, 0, r);
    matrix* Q_head_sub = submatrix(R, 0, r+buf, 0, r);
    matrix_dgemm(Q, Q_head_sub, new_X, 1.0, 0.0);

    // Switch so the QR lives in X
    double* tmp = sketch->scratch;
    sketch->scratch = sketch->X;
    sketch->X = tmp;

    free(X_big);              X_big = NULL;
    free(new_X);              new_X = NULL;
    free(Q_head_sub);         Q_head_sub = NULL;
    flattening_info_free(fi); fi = NULL;
    free(Ns);                 Ns = NULL;
    free(Q_big);              Q_big = NULL;
    free(Q);                  Q = NULL;
    if (rank == head){
        matrix_free(Q_head); Q_head = NULL;
        free(R);             R = NULL;
    }
    else{
        matrix_free(R); R = NULL;
    }
}

void perform_sketch(sketch* s, VTime* tm)
{
    multi_perform_sketch(&s, 1, tm);
}

// Gets the sketch block from the correct owner (stored in fi->f_v_block). Returns a matrix with the correct dimensions
void sendrecv_sketch_block(matrix* mat, sketch* s, flattening_info* fi, int recv_rank, int with_buf)
{
    int f_v_block = fi->f_v_block;


    MPI_tensor* ten = s->ten;
    int rank = ten->rank;
    MPI_Comm comm = ten->comm;

    int owner = s_get_owner(s, f_v_block);
    int r = (with_buf) ? s->r + s->buf : s->r;

    if (rank == recv_rank){
        if (rank == owner){
            own_submatrix_update(mat, s, f_v_block, with_buf);
        }
        else{
            matrix_wrap_update(mat, fi->f_N, r, s->scratch);
            matrix_recv(comm, mat, owner);
        }
    }
    else{
        if(rank == owner){
            own_submatrix_update(mat, s, f_v_block, with_buf);
            matrix_send(comm, mat, recv_rank);
//            free(s_mat_send);
        }
        mat->X = NULL;
    }
}

// Eats s, so be careful!
MPI_tensor* sketch_to_tensor(sketch** s_ptr)
{
    sketch* s = *s_ptr;

    MPI_tensor* sten = (MPI_tensor*) malloc(sizeof(MPI_tensor));
    MPI_tensor* ten = s->ten;
    int rank = ten->rank;
    int size = ten->comm_size;

    flattening_info* fi = flattening_info_init(ten, s->flattening, s->iscol, 0);

    // Get the schedule from the owner_partition
    int n_schedule = 0;
    int* owners = s->owner_partition;
    for (int ii = 0; ii < size; ++ii){
        int sz = owners[ii+1] - owners[ii];
        n_schedule = (n_schedule > sz) ? n_schedule : sz;
    }

    int** schedule = (int**) malloc(size * sizeof(int*));
    int* inverse_schedule = (int*) malloc(fi->f_Nblocks * sizeof(int));
    for (int ii = 0; ii < size; ++ii){
        schedule[ii] = (int*) malloc(n_schedule * sizeof(int));

        int* schedule_ii = schedule[ii];
        int jj0 = owners[ii];
        int sz = owners[ii+1] - jj0;
        for (int jj = 0; jj < n_schedule; ++jj){
            if (jj < sz){
                schedule_ii[jj] = jj+jj0;
                inverse_schedule[jj+jj0] = ii*n_schedule + jj;
            }
            else{
                schedule_ii[jj] = -1;
            }
        }
    }

    int d = fi->f_d + 1;
    int soffset = (s->iscol) ? 0 : 1;
    int offset = (s->iscol) ? 0 : s->flattening;
    int end_ind = (s->iscol) ? d-1 : 0;

    // Inherit the properties of ten
    int* n = (int*) malloc(d * sizeof(int));
    int* nps = (int*) malloc(d * sizeof(int));
    int** partitions = (int**) malloc(d * sizeof(int*));
    for (int ii = 0; ii < d-1; ++ii){
        n[ii + soffset] = ten->n[ii + offset];
        nps[ii + soffset] = ten->nps[ii + offset];
        partitions[ii + soffset] = (int*) malloc( (nps[ii+soffset] + 1) * sizeof(int));

        int* spartition_ii = partitions[ii+soffset];
        int* partition_ii = ten->partitions[ii+offset];
        for (int jj = 0; jj < nps[ii+soffset] + 1; ++jj){
            spartition_ii[jj] = partition_ii[jj];
        }
    }
    // Set the special sketched dimension properties
    n[end_ind] = s->r;
    nps[end_ind] = 1;
    partitions[end_ind] = (int*) malloc( (nps[end_ind] + 1) * sizeof(int));
    int* spartition_end = partitions[end_ind];
    spartition_end[0] = 0; spartition_end[1] = n[end_ind];


    // Get the parameters giving the subtensor locations
    double** subtensors = (double**) malloc(n_schedule * sizeof(double*));
    int* schedule_rank = schedule[rank];
    int with_buf = 0;
    long scratch_offset = 0;
    for (int ii = 0; ii < n_schedule; ++ii){
        if (schedule_rank[ii] != -1){
            matrix* X_submat = own_submatrix(s, schedule_rank[ii], with_buf);
            subtensors[ii] = s->scratch + scratch_offset;
            matrix* scratch_submat = matrix_wrap(X_submat->m, X_submat->n, subtensors[ii]);
            matrix_copy_data(scratch_submat, X_submat);
            scratch_offset = scratch_offset + X_submat->n * X_submat->m;
            free(X_submat);
            free(scratch_submat);
        }
    }
    void* parameters = p_static_init(subtensors);

    // Assigning fields
    sten->d = d;
    sten->n = n;

    sten->comm = ten->comm;
    sten->rank = ten->rank;
    sten->comm_size = ten->comm_size;

    sten->schedule = schedule;
    sten->n_schedule = n_schedule;
    sten->inverse_schedule = inverse_schedule;

    sten->partitions = partitions;
    sten->nps = nps;

    sten->current_part = -1;

    sten->f_ten = NULL;
    sten->parameters = parameters;

    sten->X_size = s->X_size;
    sten->X = s->scratch;

    sten->ind1 = (int*) malloc(d * sizeof(int));
    sten->ind2 = (int*) malloc(d * sizeof(int));
    sten->tensor_part = (int*) malloc(d * sizeof(int));
    sten->group_ranks = (int*) malloc(ten->comm_size * sizeof(int));
    sten->t_kk = (int*) malloc(d * sizeof(int));

    // Freeing sketch things
    int n_Omega = s->iscol ? ten->d - s->flattening : s->flattening;
    for (int ii = 0; ii < n_Omega; ++ii){
        matrix_free(s->Omegas[ii]); s->Omegas[ii] = NULL;
    }
    for (int ii = 0; ii < 4; ++ii){
        matrix_free(s->KRs[ii]); s->KRs[ii] = NULL;
    }
    free(s->KRs); s->KRs = NULL;
    s->ten = NULL;
    free(s->owner_partition); s->owner_partition = NULL;
    free(s->X);               s->X = NULL;
    free(s->Omegas);          s->Omegas = NULL;
    free(s->recv_buf);        s->recv_buf = NULL;
    flattening_info_free(s->fi); s->fi = NULL;
    free(s);                  *s_ptr = NULL;


    flattening_info_free(fi);

    return sten;
}