#include "SSTT.h"
#include "matrix.h"
#include "tensor.h"
#include "tt.h"
#include "sketch.h"
#include "VTime.h"

#include <stdlib.h>
#include <lapacke.h>
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <gperftools/heap-profiler.h>

void SSTT_sketch_to_train(sketch* s, tensor_train* tt, int train_ind)
{
    int head = 0;
    MPI_tensor* ten = s->ten;
    MPI_Comm comm = ten->comm;
    int iscol = s->iscol;
    int flattening = s->flattening;

    flattening_info* fi = flattening_info_init(ten, flattening, iscol, 0);

    // train stuff
    int d = ten->d;
    int n = tt->n[train_ind];
    int r1 = tt->r[train_ind];
    int r2 = tt->r[train_ind + 1];
    double* train = tt->trains[train_ind];
    matrix* train_mat = matrix_wrap(r1 * n, r2, train);
    matrix_fill_zeros(train_mat);

    int* op = s->owner_partition;
    int rank = ten->rank;

    matrix* train_submat = submatrix(train_mat, 0, 0, 0, 0);
    for (int ii = op[rank]; ii < op[rank+1]; ++ii){
        matrix* X_submat = own_submatrix(s, ii, 0);
        matrix* X_subsubmat = submatrix(X_submat, 0, 0, 0, 0);

        flattening_info_f_update(fi, ten, ii);

        int i0_train = fi->f_t_index[0];
        int sz = fi->f_t_sizes[0];

        int jj_max = 1;
        if (train_ind != 0){
            i0_train = i0_train + r1 * fi->f_t_index[1];
            jj_max = fi->f_t_sizes[1];
        }

        for (int jj = 0; jj < jj_max; ++jj){
            submatrix_update(train_submat, i0_train + jj*r1, i0_train + sz + jj*r1, 0, r2);
            submatrix_update(X_subsubmat, jj*sz, (jj+1)*sz, 0, r2);
            matrix_copy_data(train_submat, X_subsubmat);
        }

        free(X_submat);     X_submat = NULL;
        free(X_subsubmat);  X_subsubmat = NULL;
    }
    free(train_submat); train_submat = NULL;

    matrix_reshape(r1*n*r2, 1, train_mat); // Reshape so MPI_Allreduce is only called once
    matrix_allreduce(comm, train_mat);


    flattening_info_free(fi);
    free(train_mat);
    return;
}

MPI_tensor* SSTT_next_ten_init(MPI_tensor* prev_ten, tensor_train* tt, int train_ind)
{
    int r2 = tt->r[train_ind + 1];

    MPI_Comm comm = prev_ten->comm;
    int rank = prev_ten->rank;
    int size = prev_ten->comm_size;
    int is_first = (train_ind == 0);

    int d = (is_first) ? prev_ten->d : prev_ten->d - 1;
    int prev_offset = (is_first) ? 0 : 1;
    int* n = (int*) calloc(d, sizeof(int));
    int* nps = (int*) calloc(d, sizeof(int));
    int Nblocks = 1;
    n[0] = r2;
    nps[0] = 1;

    int** partitions = (int**) malloc(d * sizeof(int*));
    partitions[0] = (int*) calloc(2, sizeof(int));
    partitions[0][1] = r2;
    for (int ii = 1; ii < d; ++ii){
        n[ii] = prev_ten->n[prev_offset + ii];
        nps[ii] = prev_ten->nps[prev_offset + ii];
        Nblocks = Nblocks * nps[ii];

        partitions[ii] = (int*) calloc(nps[ii]+1, sizeof(int));
        int* partition_ii = partitions[ii];
        int* prev_partition_ii = prev_ten->partitions[prev_offset + ii];
        for (int jj = 0; jj < nps[ii]+1; ++jj){
            partition_ii[jj] = prev_partition_ii[jj];
        }
    }

    int** schedule = (int**) calloc(size, sizeof(int*));
    int n_schedule = get_schedule(schedule, nps, d, size);
    int* inverse_schedule = (int*) calloc(Nblocks, sizeof(int));
    for (int ii = 0; ii < size; ++ii){
        int* schedule_ii = schedule[ii];
        for (int jj = 0; jj < n_schedule; ++jj){
            if (schedule_ii[jj] != -1){
                inverse_schedule[schedule_ii[jj]] = ii*n_schedule + jj;
            }
        }
    }


    long* subtensor_sizes = (long*) malloc(n_schedule * sizeof(long));
    int* t_t_block = (int*) malloc(d*sizeof(int));
    long X_size = 0;
    for (int jj = 0; jj < n_schedule; ++jj){

        int t_v_block = schedule[rank][jj];

        if (t_v_block != -1){
            subtensor_sizes[jj] = 1;
            to_tensor_ind(t_t_block, t_v_block, nps, d);

            for (int kk = 0; kk < d; ++kk){
                subtensor_sizes[jj] = subtensor_sizes[jj] * (partitions[kk][t_t_block[kk]+1] - partitions[kk][t_t_block[kk]]);
            }
            X_size = X_size + subtensor_sizes[jj];
        }
        else{
            subtensor_sizes[jj] = 0;
        }
    }
    free(t_t_block);
    double* X = (double*) calloc(X_size, sizeof(double)); // callocing here, because I will probably need it to start at 0


    double* X_subtensor = X;
    double** subtensors = (double**) malloc(n_schedule * sizeof(double*));
    for (int jj = 0; jj < n_schedule; ++jj){
        subtensors[jj] = X_subtensor;
        X_subtensor = X_subtensor + subtensor_sizes[jj];
    }
    int flattening = d;
    void* parameters = p_static_init(subtensors);

    free(subtensor_sizes);




    // Assigning fields
    MPI_tensor* next_ten = (MPI_tensor*) malloc(sizeof(MPI_tensor));

    next_ten->d = d;
    next_ten->n = n;

    next_ten->comm = comm;
    next_ten->rank = rank;
    next_ten->comm_size = size;

    next_ten->schedule = schedule;
    next_ten->n_schedule = n_schedule;
    next_ten->inverse_schedule = inverse_schedule;

    next_ten->partitions = partitions;
    next_ten->nps = nps;

    next_ten->current_part = -1;

    next_ten->f_ten = NULL;
    next_ten->parameters = parameters;

    next_ten->X_size = X_size;
    next_ten->X = X;

    next_ten->ind1 = (int*) malloc(d * sizeof(int));
    next_ten->ind2 = (int*) malloc(d * sizeof(int));
    next_ten->tensor_part = (int*) malloc(d * sizeof(int));
    next_ten->group_ranks = (int*) malloc(size * sizeof(int));
    next_ten->t_kk = (int*) malloc(d * sizeof(int));

    return next_ten;
}


MPI_tensor* SSTT_next_ten(MPI_tensor* prev_ten, tensor_train* tt, int train_ind)
{
    // Get the train matrix. This is what we multiply prev_ten by!
    int n0 = tt->n[train_ind];
    int r1 = tt->r[train_ind];
    int r2 = tt->r[train_ind + 1];
    double* train = tt->trains[train_ind];
    matrix* train_mat = matrix_wrap(r1 * n0, r2, train);

    // Allocate memory for next_ten
    MPI_tensor* next_ten = SSTT_next_ten_init(prev_ten, tt, train_ind);

    // Allocate scratch memory
    long scratch_size = 1;
    for (int ii = 0; ii < next_ten->d; ++ii){
        int* partition_ii = next_ten->partitions[ii];
        int size_ii = 0;
        for (int jj = 0; jj < next_ten->nps[ii]; ++jj){
            int tmp = partition_ii[jj+1] - partition_ii[jj];
            size_ii = (size_ii > tmp) ? size_ii : tmp;
        }
        scratch_size = scratch_size * size_ii;
    }
    double* scratch = (double*) malloc(scratch_size * sizeof(double));

    // Get some indexing stuff
    int iscol = 1;
    int prev_flattening = (train_ind == 0) ? 1 : 2;
    flattening_info* prev_fi = flattening_info_init(prev_ten, prev_flattening, iscol, 0);
    flattening_info* prev_fi_tmp = flattening_info_init(prev_ten, prev_flattening, iscol, 0);

    int next_flattening = 1;
    flattening_info* next_fi = flattening_info_init(next_ten, next_flattening, iscol, 0);
    flattening_info* next_fi_tmp = flattening_info_init(next_ten, next_flattening, iscol, 0);

    // Streaming loop
    int rank = prev_ten->rank;
    int size = prev_ten->comm_size;
    int* prev_schedule_rank = prev_ten->schedule[rank];
    int* next_schedule_rank = next_ten->schedule[rank];

    MPI_Group world_group;
    MPI_Comm_group(next_ten->comm, &world_group);

    matrix* prev_mat = (matrix*) calloc(1, sizeof(matrix));
    matrix* next_mat = (matrix*) calloc(1, sizeof(matrix));
    matrix* share_mat = (matrix*) calloc(1, sizeof(matrix));

    long buf_size = 1;
    for (int ii = 0; ii < next_ten->d; ++ii){
        int sz_ii = 0;
        int* partition_ii = next_ten->partitions[ii];
        for (int jj = 0; jj < next_ten->nps[ii]; ++jj){
            int sz_ii_jj = partition_ii[jj+1] - partition_ii[jj];
            sz_ii = (sz_ii > sz_ii_jj) ? sz_ii : sz_ii_jj;
        }
        buf_size *= sz_ii;
    }
    matrix* buf = matrix_init(buf_size, 1);

    for (int ii = 0; ii < prev_ten->n_schedule; ++ii){
        // Some initialization - figuring out who owns and needs what
        int prev_block_rank = prev_schedule_rank[ii];

        int* next_t_v_blocks = (int*) malloc(size*sizeof(int));
        for (int jj = 0; jj < size; ++jj){
            int prev_block_jj = prev_ten->schedule[jj][ii];

            if (prev_block_jj != -1){
                flattening_info_update(prev_fi_tmp, prev_ten, prev_block_jj);
                int* next_t_t_block = (prev_fi_tmp->t_t_block) + prev_flattening;
                int* next_nps = (next_fi->t_nps) + 1;
                next_t_v_blocks[jj] = to_vec_ind(next_t_t_block, next_nps, next_fi->t_d - 1);
            }
            else{
                next_t_v_blocks[jj] = -1;
            }
        }

        if (prev_block_rank != -1){
            // Get prev_mat
            stream(prev_ten, prev_block_rank);
            flattening_info_update(prev_fi, prev_ten, prev_block_rank);
            matrix_wrap_update(prev_mat, prev_fi->f_N, prev_fi->s_N, get_X(prev_ten));

            // Get next_mat
            int next_t_v_block = next_t_v_blocks[rank];
            flattening_info_update(next_fi, next_ten, next_t_v_block);

            int next_owner;
            int next_epoch;
            MPI_tensor_get_owner(next_ten, next_t_v_block, &next_owner, &next_epoch);

            double beta;

            if (next_owner == rank){
                beta = 1.0;
                stream(next_ten, next_schedule_rank[next_epoch]);
                matrix_wrap_update(next_mat, next_fi->f_N, next_fi->s_N, get_X(next_ten));
            }
            else{
                beta = 0.0;
                matrix_wrap_update(next_mat, next_fi->f_N, next_fi->s_N, scratch);
            }

            // Get train submat
            int i0 = r1 * prev_fi->f_t_index[prev_flattening - 1];
            int i1 = i0 + r1 * prev_fi->f_t_sizes[prev_flattening - 1];
            submatrix_update(train_mat, i0, i1, 0, r2);
            train_mat->transpose = 1;

            // Multiply
            matrix_dgemm(train_mat, prev_mat, next_mat, 1.0, beta);
        }


        // Share
        int* already_shared = (int*) calloc(size, sizeof(int));
        int* group_ranks = prev_ten->group_ranks;
//        printf("Starting share\n");
        for (int jj = 0; jj < size; ++jj){
            int group_size = 0;
            int t_v_block_jj = next_t_v_blocks[jj];
            int owner_jj;
            int epoch_jj;
            MPI_tensor_get_owner(next_ten, t_v_block_jj, &owner_jj, &epoch_jj);
            if (owner_jj == jj){
                already_shared[jj] = 1;
            }
            else if (t_v_block_jj == -1){
                already_shared[jj] = 1;
            }
            else if(already_shared[jj] == 0){
                int owner_ind;
                int rank_ind;
                for (int kk = 0; kk < size; ++kk){
                    if (kk == rank){
                        rank_ind = group_size;
                    }

                    if (kk == owner_jj){
                        group_ranks[group_size] = kk;
                        group_size = group_size + 1;
                    }
                    else if (next_t_v_blocks[kk] == t_v_block_jj){
                        already_shared[kk] = 1;
                        group_ranks[group_size] = kk;
                        group_size = group_size + 1;
                    }
                }

                matrix* reduce_mat = NULL;
                if (owner_jj == rank){
                    flattening_info_update(next_fi_tmp, next_ten, t_v_block_jj);
                    stream(next_ten, t_v_block_jj);
                    matrix_wrap_update(share_mat, next_fi_tmp->t_N, 1, get_X(next_ten));
                    reduce_mat = share_mat;
                }
                else{
                    matrix_reshape(next_mat->n * next_mat->m, 1, next_mat);
                    reduce_mat = next_mat;
                }

                matrix_reshape(reduce_mat->n, reduce_mat->m, buf);
                matrix_group_reduce(next_ten->comm, rank, reduce_mat, buf, owner_jj, group_ranks, group_size);
            }
        }


        free(next_t_v_blocks);
        free(already_shared);
    }
    free(prev_mat);
    free(next_mat);
    free(share_mat);
    matrix_free(buf);


    free(train_mat);
    free(scratch);

    flattening_info_free(prev_fi);
    flattening_info_free(prev_fi_tmp);
    flattening_info_free(next_fi);
    flattening_info_free(next_fi_tmp);

    MPI_Group_free(&world_group);
    return next_ten;
}

void SSTT_copy_final_train(MPI_tensor* ten, tensor_train* tt)
{
    int r1 = tt->r[tt->d - 1];
    int n  = tt->n[tt->d - 1];
    matrix* train_mat = matrix_wrap(r1, n, tt->trains[tt->d - 1]);

    int rank = ten->rank;
    MPI_Comm comm = ten->comm;
    int size = ten->comm_size;

    int flattening = 1;
    int iscol = 0;
    flattening_info* fi = flattening_info_init(ten, flattening, iscol, 0);

    int Nblocks = ten->nps[1];
    int head = 0;
    for (int ii = 0; ii < Nblocks; ++ii){
        int rank_ii;
        int epoch_ii;
        MPI_tensor_get_owner(ten, ii, &rank_ii, &epoch_ii);

        flattening_info_update(fi, ten, ii);
        matrix* train_submat = submatrix(train_mat, 0, r1, fi->f_t_index[0], fi->f_t_index[0] + fi->f_N);

        if (rank_ii == rank){
            stream(ten, ii);
            matrix* ten_mat = matrix_wrap(train_submat->m, train_submat->n, get_X(ten));
            if (rank == head){
                matrix_copy_data(train_submat, ten_mat);
            }
            else{
                matrix_send(comm, ten_mat, head);
            }
            free(ten_mat);
        }
        else if (rank == head){
            matrix_recv(comm, train_submat, rank_ii);
        }

        free(train_submat);
    }


    free(train_mat);
    flattening_info_free(fi);
}


// This frees ten. So just be careful
VTime* SSTT(tensor_train* tt, MPI_tensor* ten){
    int d = ten->d;
    int buf = 2;
    int iscol = 1;
    VTime* tm = VTime_init(1 + 4*(d-1));

    char** labels = (char**) malloc(4*sizeof(char*));
    for (int ii = 0; ii < 4; ++ii){
        labels[ii] = (char*) malloc(100*sizeof(char));
    }
    strcpy(labels[0], "ii%d perform_sketch");
    strcpy(labels[1], "ii%d sketch_qr");
    strcpy(labels[2], "ii%d SSTT_sketch_to_train");
    strcpy(labels[3], "ii%d SSTT_next_ten");

    for (int ii = 0; ii < d-1; ++ii){
        tm->offset = 1 + ii * 4;
        VTime_break(tm, -1, NULL);
        int flattening = (ii == 0) ? 1 : 2;
        sketch* s = sketch_init(ten, flattening, tt->r[ii+1], buf, iscol); // Initialize sketch
//        tm->offset = 1; // This will have to change at some point
        perform_sketch(s, NULL); // Sketch
        VTime_break(tm, 0, NULL);

        sketch_qr(s); // QR
        VTime_break(tm, 1, NULL);

        SSTT_sketch_to_train(s, tt, ii); // Get the train
        VTime_break(tm, 2, NULL);

        MPI_tensor* prev_ten = ten;
        ten = SSTT_next_ten(prev_ten, tt, ii); // Perform train^T * ten
        VTime_break(tm, 3, NULL);

        sketch_free(s);
        MPI_tensor_free(prev_ten); prev_ten = NULL;

        for (int jj = 0; jj < 4; ++jj){
            sprintf(tm->labels[jj + tm->offset], labels[jj], ii);
        }
    }
    SSTT_copy_final_train(ten, tt); // copy ten into trains[d-1]

    MPI_tensor_free(ten);

    for (int ii = 0; ii < 4; ++ii){
        free(labels[ii]); labels[ii] = NULL;
    }
    free(labels); labels = NULL;

    VTime_finalize(tm);
    return tm;
}