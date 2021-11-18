
#include <lapacke.h>
#include <mpi.h>
#include <stdio.h>

#include "../include/paralleltt.h"

#include <unistd.h>

// X is r1 x n x r2, transpose is r2 x n x r1
double* train_transpose(const double* X, int r1, int n, int r2){
    double* X_T = (double*) calloc(r1*n*r2, sizeof(double));

    for (int ii = 0; ii < r1; ++ii){
        for (int jj = 0; jj < n; ++jj){
            for (int kk = 0; kk < r2; ++kk){
                X_T[kk + jj*r2 + ii*n*r2] = X[ii + jj*r1 + kk*n*r1];
            }
        }
    }

    return X_T;
}



void two_sketches_to_train(sketch* s1, sketch* s2, double** train_ptr)
{
    double* train = *train_ptr;
    int head = 0;
    if ((s1 == NULL) || (s2 == NULL)){
        sketch* s = (s1 == NULL) ? s2 : s1;
        MPI_tensor* ten = s->ten;
        MPI_Comm comm = ten->comm;
        int iscol = s->iscol;
        int flattening = s->flattening;

        flattening_info* fi = flattening_info_init(ten, flattening, iscol, 0);

        int d = ten->d;
        int n = ten->n[(iscol) ? 0 : d-1];
        int r = s->r;

        matrix_tt* train_mat = (iscol) ? matrix_tt_wrap(n, r, train) : matrix_tt_wrap(r, n, train);
        matrix_tt_fill_zeros(train_mat);

        int* op = s->owner_partition;
        int rank = ten->rank;

        for (int ii = op[rank]; ii < op[rank+1]; ++ii){
            // X submatrix
            matrix_tt* X_submat = own_submatrix(s, ii, 0);

            // train submatrix
            flattening_info_f_update(fi, ten, ii);
            int i0 = fi->f_t_index[0];
            int i1 = i0 + fi->f_t_sizes[0];

            matrix_tt* train_submat = (iscol) ? submatrix(train_mat, i0, i1, 0, r) : submatrix(train_mat, 0, r, i0, i1);
            train_submat->transpose = (iscol) ? 0 : 1;
            matrix_tt_copy_data(train_submat, X_submat);

            free(train_submat);
            free(X_submat);
        }
        matrix_tt_reduce(comm, rank, train_mat, NULL, head);


        flattening_info_free(fi);
        free(train_mat);
        return;
    }


    int iscol = s1->iscol ? 1 : 0;
    int iscol2 = s2->iscol ? 1 : 0;

    if (iscol != iscol2){
        printf("two_sketches_to_train: Cannot convert a column and a row to a train\n");
        return;
    }

    int flattening1 = s1->flattening;
    int flattening2 = s2->flattening;

    // We want s2 to be the taller sketch, so that we only communicate s1 if necessary (which should be cheaper!)
    if (((iscol) && (flattening1 > flattening2)) || (!(iscol) && (flattening2 > flattening1))){
        sketch* tmp = s1;
        s1 = s2;
        s2 = tmp;

        flattening1 = s1->flattening;
        flattening2 = s2->flattening;
    }

    // We want the flattening dimensions to be one off of each other
    MPI_tensor* ten = s1->ten;
    flattening_info* fi1 = flattening_info_init(ten, flattening1, iscol, 0);
    flattening_info* fi2 = flattening_info_init(ten, flattening2, iscol, 0);
    if (fi2->f_d - fi1->f_d != 1){
        printf("two_sketches_to_train: The difference between the two flattening dimensions is not correct!\n");
        return;
    }

    MPI_Comm comm = ten->comm;
    int size = ten->comm_size;
    int rank = ten->rank;

    int r1 = s1->r;
    int r2 = s2->r;
    int n_index = (iscol) ? flattening1 : flattening2;
    int n = ten->n[n_index];
    matrix_tt* train_mat = matrix_tt_wrap(r1, n*r2, train);
    matrix_tt_fill_zeros(train_mat);


    // Loop over all blocks of fi2
    matrix_tt* s_mat1 = (matrix_tt*) calloc(1, sizeof(matrix_tt));
    for (int block2 = 0; block2 < fi2->f_Nblocks; ++block2){
        flattening_info_f_update(fi2, ten, block2);
        int owner2 = s_get_owner(s2, block2);


        // Get the corresponding block1
        int* f_t_block1 = fi2->f_t_block + ((iscol) ? 0 : 1);
        int* f_nps1 = fi2->f_nps + ((iscol) ? 0 : 1);
        int block1 = 0;
        for (int ii = (fi1->f_d) - 1; ii >= 0; --ii){
            block1 = block1*f_nps1[ii] + f_t_block1[ii];
        }

        flattening_info_f_update(fi1, ten, block1);
        int owner1 = s_get_owner(s1, block1);
        sendrecv_sketch_block(s_mat1, s1, fi1, owner2, 0);

        if (rank == owner2){

            // Get submatrix of s2
            matrix_tt* s_mat2 = own_submatrix(s2, block2 , 0);

            // Multiply
            matrix_tt* train_submat = submatrix(train_mat, 0, 0, 0, 0);
            matrix_tt* s_submat2 = submatrix(s_mat2, 0, 0, 0, 0);
            s_submat2->transpose = (iscol) ? 0 : 1;
            for (int kk = 0; kk < r2; ++kk){
                int jj0 = (fi2->f_t_index[(iscol) ? (fi2->f_d) - 1 : 0]) + kk * n;
                int sz = fi2->f_t_sizes[(iscol) ? (fi2->f_d) - 1 : 0];
                submatrix_update(train_submat, 0, r1, jj0, jj0+sz);

                s_submat2->X = s_mat2->X + s_mat2->offset + kk*(s_mat2->lda);
                if (iscol){
                    s_submat2->m = s_mat1->m;
                    s_submat2->lda = s_mat1->m;
                    s_submat2->n = sz;
                }
                else{
                    s_submat2->m = sz;
                    s_submat2->lda = sz;
                    s_submat2->n = s_mat1->m;
                }

                s_mat1->transpose = 1;
                matrix_tt_dgemm(s_mat1, s_submat2, train_submat, 1.0, 1.0);
            }
            free(train_submat);
            free(s_mat2);
            free(s_submat2);
        }
    }
    free(s_mat1);

    matrix_tt_reduce(comm, rank, train_mat, NULL, head);

    // If it was a row sketch, we need to transpose
    if ((!iscol) && (rank == head)){
        double* train_cp = train_transpose(train, r1, n, r2);
        *train_ptr = train_cp;

        matrix_tt_free(train_mat); train_mat = NULL;
    }
    else{
        free(train_mat); train_mat = NULL;
    }

    flattening_info_free(fi1);
    flattening_info_free(fi2);
}


void PSTT2_final_train(tensor_train* tt, sketch** sketches, int mid)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int head = 0;

    sketch* s1 = sketches[mid-1];
    int r1 = s1->r;
    sketch* s2 = sketches[mid];
    int r2 = s2->r;
    int n = tt->n[mid];



    MPI_tensor* ten = s1->ten;
    MPI_Comm comm = ten->comm;
    int size = ten->comm_size;
    int* schedule_rank = ten->schedule[rank];



    flattening_info* fi1 = flattening_info_init(ten, s1->flattening, s1->iscol, 0);
    flattening_info* fi2 = flattening_info_init(ten, s2->flattening, s2->iscol, 0);

    matrix_tt* train_mat = matrix_tt_wrap(r1*n, r2, tt->trains[mid]);
    matrix_tt_fill_zeros(train_mat);



    int tmp_n = 1;
    int* mid_partition = ten->partitions[mid];
    for (int ii = 0; ii < ten->nps[mid]; ++ii){
        int candidate = mid_partition[ii+1] - mid_partition[ii];
        tmp_n = (candidate > tmp_n) ? candidate : tmp_n;
    }
    matrix_tt* tmp = matrix_tt_init(r1, tmp_n);

    matrix_tt* recv_mat = (matrix_tt*) calloc(1, sizeof(matrix_tt));
    matrix_tt* s_mat1 = (matrix_tt*) calloc(1, sizeof(matrix_tt));
    matrix_tt* s_mat2 = (matrix_tt*) calloc(1, sizeof(matrix_tt));
    matrix_tt* s_submat2 = (matrix_tt*) calloc(1, sizeof(matrix_tt));
    matrix_tt* subtensor_mat = (matrix_tt*) calloc(1, sizeof(matrix_tt));
    // Loop over schedule and stream
    for (int ii = 0; ii < ten->n_schedule; ++ii){
        int block = schedule_rank[ii];
        stream(ten, block);

        // Get the correct sketch matrices for multiplication
        for (int jj = 0; jj < size; ++jj){
            int* schedule_jj = ten->schedule[jj];
            int block_jj = schedule_jj[ii];

            if (block_jj != -1){
                // NOTE: Valgrind says I have a memory leak with these allocations.
                //       I think it's just confused by MPI, but maybe it's worth checking out
                flattening_info_update(fi1, ten, block_jj);
                sendrecv_sketch_block(recv_mat, s1, fi1, jj, 0);
                if (recv_mat->X != NULL){
                    matrix_tt* swtch = recv_mat;
                    recv_mat = s_mat1;
                    s_mat1 = swtch;
                }


                flattening_info_update(fi2, ten, block_jj);
                sendrecv_sketch_block(recv_mat, s2, fi2, jj, 0);
                if (recv_mat->X != NULL){
                    matrix_tt* swtch = recv_mat;
                    recv_mat = s_mat2;
                    s_mat2 = swtch;
                }
            }
        }


        flattening_info_update(fi1, ten, block);
        flattening_info_update(fi2, ten, block);

        // Multiply
        if (block != -1){
            int jj0 = fi1->t_t_index[mid];
            int sz = fi1->t_t_sizes[mid];
            submatrix_update(tmp, 0, r1, 0, sz);
            matrix_tt_wrap_update(subtensor_mat, fi1->f_N, fi1->s_N, get_X(ten));
            s_mat1->transpose = 1;
            submatrix_update(train_mat, jj0*r1, (jj0 + sz)*r1, 0, r2);
            submatrix_update(subtensor_mat, 0, 0, 0, 0);

            s_submat2->transpose = s_mat2->transpose;
            s_submat2->offset = 0;
            s_submat2->lda = s_mat2->lda;
            s_submat2->X_size = s_mat2->X_size;
            s_submat2->X = s_mat2->X + s_mat2->offset;


            for (int jj = 0; jj < fi2->f_N; ++jj){
                matrix_tt_reshape(r1, sz, tmp);
                submatrix_update(subtensor_mat, 0, fi1->f_N, jj*sz, (jj+1)*sz);

                matrix_tt_dgemm(s_mat1, subtensor_mat, tmp, 1.0, 0.0);
                matrix_tt_reshape(r1*sz, 1, tmp);


                submatrix_update(s_submat2, jj, jj+1, 0, r2);
                matrix_tt_dgemm(tmp, s_submat2, train_mat, 1.0, 1.0);
            }
        }

    }
    free(recv_mat);
    free(s_mat1);
    free(s_mat2);
    free(s_submat2);
    free(subtensor_mat);

    flattening_info_free(fi1);
    flattening_info_free(fi2);

    submatrix_update(train_mat, 0, n*r1, 0, r2);
    matrix_tt_reduce(comm, rank, train_mat, NULL, head);
    matrix_tt_free(tmp);

    free(train_mat);
}

int PSTT2_get_mid(MPI_tensor* ten, int mid)
{
    int d = ten->d;
    if ((mid > 0) && (mid < d)){
        return mid;
    }

    int* n = ten->n;

    long n_left = 1;
    long n_right = 1;
    for (int ii = 0; ii < d; ++ii){
        n_right = (long) n_right * n[ii];
    }

    for (int ii = 0; ii < d; ++ii){
        n_left = (long) n_left * n[ii];
        n_right = (long) n_right / n[ii];
        if ( (mid == -1) && (n_right < n_left) ){
            mid = ii;
        }
    }

    // Just in case, but this should never actually happen
    if (mid == -1){
        printf("You entered a weird tensor (probably n_i = 1 identically), setting mid = d-1\n");
        mid = d-1;
    }

    return mid;
}


void PSTT2(tensor_train* tt, MPI_tensor* ten, int mid)
{
    mid = PSTT2_get_mid(ten, mid);
    int rank = ten->rank;
    int BUF = 2;
    int* r = tt->r;
    int d = ten->d;


    // Create sketches
    sketch** sketches = (sketch**) calloc(d-1, sizeof(sketch*));
    for (int ii = 0; ii < d-1; ++ii){
        int iscol = (ii < mid) ? 1 : 0; // Column sketch below mid, row sketch above it
        sketches[ii] = sketch_init(ten, ii+1, r[ii+1], BUF, iscol);
    }

    // Sketch the tensor
    multi_perform_sketch(sketches, d-1);


    for (int ii = 0; ii < d-1; ++ii){
        sketch_qr(sketches[ii]);
    }

    // multiply out the sketches
    for (int ii = 0; ii < mid; ++ii){
        sketch* s1 = (ii == 0) ? NULL : sketches[ii-1];
        sketch* s2 = sketches[ii];
        two_sketches_to_train(s1, s2, tt->trains + ii);
    }

    for (int ii = mid; ii < d-1; ++ii){
        sketch* s1 = (ii == d-2) ? NULL : sketches[ii+1];
        sketch* s2 = sketches[ii];
        two_sketches_to_train(s1, s2, tt->trains + ii + 1);
    }

    // Get the middle train
    PSTT2_final_train(tt, sketches, mid);

    // free
    for (int ii = 0; ii < d-1; ++ii){
        sketch_free(sketches[ii]); sketches[ii] = NULL;
    }
    free(sketches);
}

void PSTT2_onepass_final_train(tensor_train* tt, sketch** sketches, int mid)
{
    sketch* sketch_mid = sketches[mid];
    sketch* Q_right = sketches[mid+1];
    sketch** Q_left = sketches + mid - 1; //Note that we are using the pointer here to feed into sketch_to_tensor

    // Multiply the middle not-QRed sketch against the sketch to its right
    int p = sketch_mid->r; // this is r_left + buf
    int r_left = tt->r[mid];
    int n_mid = tt->n[mid];
    int r_right = tt->r[mid+1];
    double* b = (double*) malloc(p*n_mid*r_right*sizeof(double)); // Named for the Ax=b solve that we will do later
    two_sketches_to_train(sketch_mid, Q_right, &b);

    MPI_tensor* ten_left = sketch_to_tensor(Q_left);

    sketch* sketch_left = sketch_init_with_Omega(ten_left, ten_left->d - 1, p, 0, 0, sketch_mid->Omegas); // buf = 0, is_col = 0
    perform_sketch(sketch_left);
    double* A = (double*) calloc(p * r_left, sizeof(double));


    two_sketches_to_train(sketch_left, NULL, &A); // This reduces the matrix A

    // Solve least squares
    matrix_tt* A_mat = matrix_tt_wrap(p, r_left, A);
    matrix_tt* b_mat = matrix_tt_wrap(p, n_mid*r_right, b);
    matrix_tt* x_mat = matrix_tt_wrap(r_left, n_mid*r_right, tt->trains[mid]);

    int head = 0;
    if (ten_left->rank == head){
        matrix_tt_dgels(x_mat, A_mat, b_mat);
    }

    // Free
    sketch_free(sketch_left); sketch_left = NULL;
    MPI_tensor_free(ten_left); ten_left = NULL;
    matrix_tt_free(A_mat); A_mat = NULL;
    matrix_tt_free(b_mat); b_mat = NULL;
    free(x_mat);
}

void PSTT2_onepass(tensor_train* tt, MPI_tensor* ten, int mid)
{
    // Number of times in PSTT2: 5
    // Number of times in multi_perform_sketch: 3
    mid = PSTT2_get_mid(ten, mid);
    int rank = ten->rank;
    int BUF = 2;
    int* r = tt->r;
    int d = ten->d;


    // Create sketches
    sketch** sketches = (sketch**) calloc(d, sizeof(sketch*));
    for (int ii = 0; ii < mid; ++ii){
        int iscol = 1; // Column sketch below mid, row sketch above it
        sketches[ii] = sketch_init(ten, ii+1, r[ii+1], BUF, iscol);
    }

    // putting the BUF into the rank here, so that two_sketches_to_train and perform_sketch don't get confused in the future
    sketches[mid] = sketch_init(ten, mid, r[mid] + BUF, 0, 0);

    for (int ii = mid + 1; ii < d; ++ii){
        int iscol = 0;
        sketches[ii] = sketch_init(ten, ii, r[ii], BUF, iscol);
    }

    // Sketch the tensor
    multi_perform_sketch(sketches, d);

    // Find orthogonal basis
    for (int ii = 0; ii < mid; ++ii){
        sketch_qr(sketches[ii]);
    }
    for (int ii = mid+1; ii < d; ++ii){
        sketch_qr(sketches[ii]);
    }

    // multiply out the sketches
    for (int ii = 0; ii < mid; ++ii){
        sketch* s1 = (ii == 0) ? NULL : sketches[ii-1];
        sketch* s2 = sketches[ii];
        two_sketches_to_train(s1, s2, tt->trains + ii);
    }

    for (int ii = mid+1; ii < d; ++ii){
        sketch* s1 = (ii == d-1) ? NULL : sketches[ii+1];
        sketch* s2 = sketches[ii];
        two_sketches_to_train(s1, s2, tt->trains + ii);
    }

    // Get the middle train
    PSTT2_onepass_final_train(tt, sketches, mid);

    // free
    for (int ii = 0; ii < d; ++ii){
        if(sketches[ii] != NULL){
            sketch_free(sketches[ii]); sketches[ii] = NULL;
        }
    }
    free(sketches);
}
