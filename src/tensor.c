#include "tensor.h"
#include "matrix.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gperftools/heap-profiler.h>

#ifndef HEAD
#define HEAD (int) 0
#endif


// Tensor ind is an array of indices, s.t. we think of X as
// X[vec_ind] = X[tensor_ind[0], tensor_ind[1], ..., tensor_ind[d-1]] (this would work in MATLAB!)
void to_tensor_ind(int* tensor_ind, long vec_ind, const int* n, int d)
{
    for (int ii = 0; ii < d; ++ii){
        tensor_ind[ii] = vec_ind % n[ii];
        vec_ind = vec_ind / n[ii];
    }
}

long to_vec_ind(const int* tensor_ind, const int* n, int d)
{
    if (d == 0){
        return (long) 0;
    }

    long vec_ind = (long) tensor_ind[d-1];
    for (int ii = d-2; ii >= 0; --ii){
        vec_ind = vec_ind*n[ii] + tensor_ind[ii];
    }
    return vec_ind;
}


long product(const int* n, int d)
{
    long N = 1;
    for (int ii = 0; ii<d; ++ii){
        N = N * (n[ii]);
    }

    return N;
}



int validate_partition(const long* partition, long n, int np)
{
    int not_valid = 0;
    if (partition[0]!=0){
        not_valid = 1;
    }
    if (partition[np]!=n){
        not_valid = 1;
    }
    for (int ii = 0; ii<np; ++ii){
        if (partition[ii+1] < partition[ii]){
            not_valid = 1;
        }
    }

    if (not_valid) {
        printf("np = %d, n = %ld\n",np, n);
        printf("partition = [");
        for (int ii = 0; ii < np; ++ii){
            printf("%ld, ", partition[ii]);
        }
        printf("%ld] is not valid for n=%ld\n",partition[np],n);
    }
    return not_valid;
}

int get_mid(int d, const int* n, int mid)
{
    if((mid > 0) && (mid < d)){
        return mid;
    }

    mid = -1;

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


// n is the side length of the middle matrix to be partitioned
int* get_partition(int np, int n)
{
    int* partition = (int*) malloc(sizeof(int)*(np+1));
    for (int ii = 0; ii < np+1; ++ii){
        partition[ii] = ceil((double) ii * n / np);
    }

    return partition;
}

long get_X_size(int** partitions, int* nps, int d)
{
    long X_size = 1;
    for (int ii = 0; ii < d; ++ii){
        long dimension_size = 0;

        int* partition_ii = partitions[ii];
        int np = nps[ii];
        for (int jj = 0; jj < np; ++jj){
            long candidate = partition_ii[jj+1] - partition_ii[jj];
            if (candidate > dimension_size){
                dimension_size = candidate;
            }
        }
        X_size = X_size * dimension_size;
    }

    return X_size;
}

int get_schedule(int** schedule, const int* nps, int d, int comm_size){
    long Np = product(nps, d);

    int n_schedule = ((Np-1)/comm_size) + 1;

    // I am partitioning the vector [0, 1, ..., Np-1] into comm_size chunks to get the schedule. So, I can
    // conveniently use get_partition!
    int* schedule_partition = get_partition(comm_size, Np);

    int part = 0;
    for (int ii = 0; ii < comm_size; ++ii){
        int n_ii = (int) schedule_partition[ii+1] - schedule_partition[ii];
        schedule[ii] = (int*) malloc(sizeof(int)*n_schedule);
        int* schedule_ii = schedule[ii];

        for (int jj = 0; jj < n_schedule; ++ jj){
            if (jj < n_ii){
                schedule_ii[jj] = part;
                part = part + 1;
            }
            else{
                schedule_ii[jj] = -1;
            }
        }
    }

    free(schedule_partition);

    return n_schedule;
}


MPI_tensor* MPI_tensor_init(int d, const int* n, int* nps, MPI_Comm comm,
                            void (*f_ten)(double* restrict, int*, int*, const void*), void* parameters)
{
    MPI_tensor* ten = (MPI_tensor*) malloc(sizeof(MPI_tensor));

    ten->comm = comm;
    MPI_Comm_rank(comm, &(ten->rank));
    MPI_Comm_size(comm, &(ten->comm_size));

    ten->d = d;
    ten->n = (int*) malloc(sizeof(int)*d);
    ten->nps = (int*) malloc(sizeof(int)*d);
    for (int ii = 0; ii<d; ++ii){
        ten->n[ii] = n[ii];
        ten->nps[ii] = nps[ii];
    }
    long N = product(n, d);

    ten->partitions = (int**) malloc(d*sizeof(int*));
    for (int ii = 0; ii < d; ++ii){
        ten->partitions[ii] = get_partition(nps[ii], n[ii]);
    }

    ten->current_part = -1;

    ten->f_ten = f_ten;
    ten->parameters = parameters;

    ten->X_size = get_X_size(ten->partitions, nps, d);
    ten->X = (double*) malloc( sizeof(double) * ten->X_size );

    ten->schedule = (int**) malloc(sizeof(int*) * ten->comm_size);
    ten->n_schedule = get_schedule(ten->schedule, nps, d, ten->comm_size);
    ten->inverse_schedule = NULL;

    ten->ind1 = (int*) malloc(d * sizeof(int));
    ten->ind2 = (int*) malloc(d * sizeof(int));
    ten->tensor_part = (int*) malloc(d * sizeof(int));
    ten->group_ranks = (int*) malloc(ten->comm_size * sizeof(int));
    ten->t_kk = (int*) malloc(d * sizeof(int));

    return ten;
}

void MPI_tensor_free(MPI_tensor* ten)
{
    if (ten->f_ten == NULL){
        p_static_free(ten->parameters); ten->parameters = NULL;
    }

    for (int ii = 0; ii < ten->comm_size; ++ii){
        free(ten->schedule[ii]); ten->schedule[ii] = NULL;
    }

    for (int ii = 0; ii < ten->d; ++ii){
        free(ten->partitions[ii]); ten->partitions[ii] = NULL;
    }

    if (ten->inverse_schedule != NULL){
        free(ten->inverse_schedule); ten->inverse_schedule = NULL;
    }

    free(ten->schedule);    ten->schedule = NULL;
    free(ten->X);           ten->X = NULL;
    free(ten->n);           ten->n = NULL;
    free(ten->partitions);  ten->partitions = NULL;
    free(ten->nps);         ten->nps = NULL;
    free(ten->ind1);        ten->ind1 = NULL;
    free(ten->ind2);        ten->ind2 = NULL;
    free(ten->tensor_part); ten->tensor_part = NULL;
    free(ten->group_ranks); ten->group_ranks = NULL;
    free(ten->t_kk);        ten->t_kk = NULL;
    free(ten);
}

void* p_static_init(double** subtensors)
{
    p_static* parameters = (p_static*) malloc(sizeof(p_static));
    parameters->subtensors = subtensors;
    return (void*) parameters;
}

void p_static_free(void* parameters)
{
    p_static* p_casted = (p_static*) parameters;
    free(p_casted->subtensors); p_casted->subtensors = NULL;
    free(p_casted);
}



void stream(MPI_tensor* ten, int part)
{
    if (part != -1){
        void (*f_ten)(double* restrict, int*, int*, const void*) = ten->f_ten;

        if (f_ten != NULL){
            int d = ten->d;
//            int* ind1 = (int*) calloc(d, sizeof(int)); // lower corner of the subtensor
//            int* ind2 = (int*) calloc(d, sizeof(int)); // upper corner of the subtensor
//            int* tensor_part = (int*) calloc(d, sizeof(int)); // the block-tensor index
            int* ind1 = ten->ind1;
            int* ind2 = ten->ind2;
            int* tensor_part = ten->tensor_part;

            to_tensor_ind(tensor_part, part, ten->nps, d);

            for (int ii = 0; ii < d; ++ii){
                int* partition = ten->partitions[ii];
                ind1[ii] = partition[tensor_part[ii]];
                ind2[ii] = partition[tensor_part[ii]+1];
            }

            (*f_ten)(ten->X, ind1, ind2, ten->parameters);

//            free(ind1);
//            free(ind2);
//            free(tensor_part);
        }
    }
    ten->current_part = part;
}

void MPI_tensor_get_owner(const MPI_tensor* ten, int t_v_block, int *rank, int *epoch)
{
    int Nblocks = 1;
    for (int ii = 0; ii < ten->d; ++ii){
        Nblocks = Nblocks * ten->nps[ii];
    }

    if ((t_v_block < 0) || (t_v_block >= Nblocks)){
        *rank = -1;
        *epoch = -1;
        return;
    }

    if (ten->inverse_schedule == NULL){
        printf("inverse_schedule has not been assigned! Cannot use MPI_tensor_get_owner without it\n");
        *rank = -1;
        *epoch = -1;
        return;
    }

    int info = ten->inverse_schedule[t_v_block];
    *rank = info / ten->n_schedule;
    *epoch = info % ten->n_schedule;
}

double* get_X(const MPI_tensor* ten){
    double* X = NULL;
    if (ten->f_ten != NULL){
        X = ten->X;
    }
    else{
        int owner_rank;
        int epoch;
        MPI_tensor_get_owner(ten, ten->current_part, &owner_rank, &epoch);
        if (ten->rank == owner_rank){
            p_static* p_casted = (p_static*) ten->parameters;
            X = p_casted->subtensors[epoch];
        }
    }
    return X;
}



// Use flattening = -1 to print everything but the actual entries
void MPI_tensor_print(const MPI_tensor* ten, int flattening)
{
    int d = ten->d;
    int* n = ten->n;
    MPI_Comm comm = ten->comm;
    int rank = ten->rank;
    int comm_size = ten->comm_size;
    int** schedule = ten->schedule;
    int n_schedule = ten->n_schedule;
    int* inverse_schedule = ten->inverse_schedule;
    int** partitions = ten->partitions;
    int* nps = ten->nps;
    int current_part = ten->current_part;
    double* X = get_X(ten);
    long X_size = ten->X_size;

//    MPI_Barrier(comm);
    if (rank == HEAD){
//        printf("rank%d The tensor has d = %d, n = [", rank, d);
//        for (int ii = 0; ii<d-1; ++ii){
//            printf("%d, ", n[ii]);
//        }
//        printf("%d]\n", n[d-1]);
//
//        printf("\nrank%d partitions:\n", rank);
//        for (int ii = 0; ii < d; ++ii){
//            int np = nps[ii];
//            int* partition = partitions[ii];
//            printf("rank%d nps[%d] = %d, partitions[%d] = [%d", rank, ii, np, ii, partition[0]);
//            for (int jj = 1; jj < np+1; ++jj){
//                printf(", %d", partition[jj]);
//            }
//            printf("]\n");
//        }
//
//        printf("rank%d n_schedule = %d, schedule = \n", rank, n_schedule);
//        for (int ii = 0; ii < n_schedule; ++ii){
//            printf("rank%d [", rank);
//            for (int jj = 0; jj < comm_size-1; ++jj){
//                printf("%d, ", schedule[jj][ii]);
//            }
//            printf("%d]\n", schedule[comm_size-1][ii]);
//        }
//
//        if (inverse_schedule != NULL){
//            printf("rank%d inverse_schedule = [", rank);
//            int Nblocks = 1;
//            for (int ii = 0; ii < d; ++ii){
//                Nblocks = Nblocks * nps[ii];
//            }
//            for (int ii = 0; ii < Nblocks-1; ++ii){
//                printf("%d, ", inverse_schedule[ii]);
//            }
//            printf("%d]\n", inverse_schedule[Nblocks-1]);
//        }
//
//        printf("rank%d current_part = %d, X_size = %ld\n", rank, current_part, X_size);
//
//        if ((flattening >= 0) && (flattening <= d)){
//            printf("rank%d The %d-th flattening of ten->X is: \n", rank, flattening);
//        }
    }

    if ((flattening >= 0) && (flattening <= d)){
        for (int ii = 0; ii<comm_size; ++ii){
            MPI_Barrier(comm);
            if (ii == rank){
                printf("rank%d:\n",rank);
                int n1 = 1;
                int n2 = 1;
                int* tensor_part = (int*) calloc(d, sizeof(int));
                to_tensor_ind(tensor_part, current_part, nps, d);
                for (int ii = 0; ii < d; ++ii){
                    int* partition_ii = partitions[ii];
                    if (ii < flattening){
                        n1 = n1*(partition_ii[tensor_part[ii] + 1] - partition_ii[tensor_part[ii]]);
                    }
                    else{
                        n2 = n2*(partition_ii[tensor_part[ii] + 1] - partition_ii[tensor_part[ii]]);
                    }
                }

//                int part_1 = current_part % np_1;
//                int part_2 = current_part / np_1;
//                int n_part = (int) partition_1[part_1 + 1] - partition_1[part_1];
//                int m_part = (int) partition_2[part_2 + 1] - partition_2[part_2];
                matrix* A = matrix_wrap(n1, n2, get_X(ten));
                matrix_print(A, 1);
                free(tensor_part); tensor_part = NULL;
                free(A); A = NULL;
            }
        }
    }
}
