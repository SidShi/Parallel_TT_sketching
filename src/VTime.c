//#include <time.h>
#include <string.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <gperftools/heap-profiler.h>

#include "../include/paralleltt.h"

#define STRMAX 100

VTime* VTime_init(int n_t){
    if (n_t < 1){
        printf("VTime_init: n_t must be at least 1 to store the total time!");
        return NULL;
    }

    VTime* tm = (VTime*) malloc(sizeof(VTime));

    tm->t = (double*) calloc(n_t, sizeof(double));
    tm->labels = (char**) calloc(n_t, sizeof(char*));
    for (int ii = 0; ii < n_t; ++ii){
        tm->labels[ii] = (char*) calloc(STRMAX, sizeof(char));
    }
    tm->n_t = n_t;
    tm->offset = 1;

    tm->t_init = MPI_Wtime();
    tm->t_break = tm->t_init;
    tm->t_tmp = 0.0;
    tm->t_final = 0.0;

    return tm;
}

void VTime_set_label(VTime* tm, int ind, char* label){
    if (label){
        strcpy(tm->labels[ind + tm->offset], label);
    }
}

void VTime_break(VTime* tm, int ind, char* label){
    if(tm){
        tm->t_tmp = MPI_Wtime();
        if ((ind >= 0) && (ind + tm->offset < tm->n_t)){
            tm->t[ind + tm->offset] += tm->t_tmp - tm->t_break;
            if (tm->labels[ind + tm->offset][0]==0){
                VTime_set_label(tm, ind, label);
            }
        }
        tm->t_break = tm->t_tmp;
//        if (tm->labels[ind + tm->offset][0]!=0){
//            printf("%s\n", tm->labels[ind+tm->offset]);
//        }
//        else{
//            printf("Setting ind=%d\n", ind);
//        }
    }
}

void VTime_finalize(VTime* tm){
    tm->t_final = MPI_Wtime();
    tm->t[0] = tm->t_final - tm->t_init;
    strcpy(tm->labels[0], "Total time");
}

void VTime_free(VTime* tm){

    for (int ii = 0; ii < tm->n_t; ++ii){
        free(tm->labels[ii]); tm->labels[ii] = NULL;
    }

    free(tm->t);      tm->t = NULL;
    free(tm->labels); tm->labels = NULL;
    free(tm);
}

void VTime_print(VTime* tm){
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (int ii = 0; ii < tm->n_t; ++ii){
        printf("r%d %s: %e\n", rank, tm->labels[ii], tm->t[ii]);
    }
}


VTime_gathered* VTime_gather(VTime* tm, MPI_Comm comm)
{
    int head = 0;
    int size;
    int rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    int n_t = tm->n_t;
    VTime_gathered* tms = NULL;
    if (rank == head){
        tms = (VTime_gathered*) malloc(sizeof(VTime_gathered));
        tms->labels = (char**) calloc(n_t, sizeof(char*));
        for (int ii = 0; ii < n_t; ++ii){
            tms->labels[ii] = (char*) calloc(STRMAX, sizeof(char));
            strcpy(tms->labels[ii], tm->labels[ii]);
        }

        tms->n_t = n_t;
        tms->size = size;
        tms->t = (double*) calloc(n_t * size, sizeof(double));
        MPI_Gather(tm->t, n_t, MPI_DOUBLE, tms->t, n_t, MPI_DOUBLE, head, comm);
    }
    else{
        MPI_Gather(tm->t, n_t, MPI_DOUBLE, NULL, n_t, MPI_DOUBLE, head, comm);
    }

    return tms;
}


void VTime_gathered_free(VTime_gathered* tms)
{
    for (int ii = 0; ii < tms->n_t; ++ii){
        free(tms->labels[ii]); tms->labels[ii] = NULL;
    }
    free(tms->labels); tms->labels = NULL;
    free(tms->t); tms->t = NULL;

    free(tms);
}