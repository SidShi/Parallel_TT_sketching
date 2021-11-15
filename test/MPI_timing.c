//#include "readArray.h"
//#include "tensor_train.h"

#include "matrix.h"
#include "tensor.h"
#include "sketch.h"
#include "tt.h"
#include "VTime.h"
#include "readArray_MPI.h"
#include "PSTT.h"
#include "my_io.h"
#include "SSTT.h"

#include <omp.h>

#include <stdio.h>
#include <cblas.h>
#include <lapacke.h>
#include <math.h>
#include <string.h>
#include <gperftools/heap-profiler.h>

#ifndef HEAD
#define HEAD ((int) 0)
#endif

#ifndef FILE_BUF
#define FILE_BUF ((int) 256)
#endif

VTime_gathered* get_TT(tensor_train* tt, MPI_tensor* ten, int mid, int solvetype, int *heap_profile, char* path, long* total_alloc_ptr);



void MyHeapProfilerStart(int heap_profile, char* path, char* prefix, int rank)
{
    if (heap_profile){
        char* profile_path = (char*) calloc(FILE_BUF, sizeof(char));
        snprintf(profile_path, FILE_BUF, "%s%s%d", path, prefix, rank);
        HeapProfilerStart(profile_path);
    }
}

void MyHeapProfilerStop(int* heap_profile, char* path, char* prefix, int rank, long* total_alloc_ptr)
{
    if (*heap_profile){
        HeapProfilerDump("");
        HeapProfilerStop();

        char file_name[256];
        sprintf(file_name, "%s%s%d.000%d.heap", path, prefix, rank, *heap_profile);
        FILE *f = fopen(file_name, "r");
        while(f){
            *heap_profile = *heap_profile + 1;
            sprintf(file_name, "%s%s%d.000%d.heap", path, prefix, rank, *heap_profile);
            f = fopen(file_name, "r");
        }
        sprintf(file_name, "%s%s%d.000%d.heap", path, prefix, rank, *heap_profile - 1);
        f = fopen(file_name, "r");

        char total_alloc_str[256];
        char c = fgetc(f);
        while( c != '['){
            c = fgetc(f);
        }
        while( c != ':'){
            c = fgetc(f);
        }
        c = fgetc(f);
        int ii = 0;
        while( c != ']'){
            if (c != ' '){
                total_alloc_str[ii] = c;
                ii = ii+1;
            }
            c = fgetc(f);
        }
        total_alloc_str[ii] = '\0';


        int info = sscanf(total_alloc_str, "%ld", total_alloc_ptr);
//        printf("total_alloc_str = %s, total_alloc = %ld\n", total_alloc_str, *total_alloc_ptr);



//        long MB = 1;
//        for (int ii = 0; ii < 20; ++ii){
//            MB = MB * 2;
//        }
//        printf("size = %f MB\n ", (double) *total_alloc_ptr / MB);
    }
}

long* gather_heap_data(int heap_profile, long* total_allocs, int size, int rank, MPI_Comm comm)
{

    int head = 0;
    long* total_allocs_matrix = NULL;

    if (heap_profile){
        if (rank == head){
            total_allocs_matrix = (long*) calloc((size + 1)*3, sizeof(long));
        }
        MPI_Gather(total_allocs, 3, MPI_LONG, total_allocs_matrix + 3, 3, MPI_LONG, head, comm);

        for (int ii = 0; ii < size; ++ii){
            for (int jj = 0; jj < 3; ++jj){
                total_allocs[jj + ii*3];
            }
        }

        if (rank == head){
            for (int jj = 1; jj < size + 1; ++jj){
                for (int ii = 0; ii < 3; ++ii){
                    total_allocs_matrix[ii] += total_allocs_matrix[ii + jj*3];
                }
            }
            long MB = 1;
            for (int ii = 0; ii < 20; ++ii){
                MB = MB*2;
            }
            printf("TOTAL ALGORITHM SIZE = %f\n", (double) total_allocs_matrix[2]/MB);
        }
    }

    return total_allocs_matrix;
}


//Input:
// argv[1] = solvetype
// argv[2] = tensortype
// -d      = dimension
// -n      = sizes
// -r      = ranks
// -p      = path
// -e      = nps
// -m      = mid (optional)
// -h      = heap_profile
// -M      = M for gaussian bumps (only for tensortype == 2)
// -g      = gamma for gaussian bumps (only for tensortype == 2)
int main (int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int world_rank;
    int world_size;
    MPI_Comm comm_world = MPI_COMM_WORLD;
    MPI_Comm_rank(comm_world, &world_rank);
    MPI_Comm_size(comm_world, &world_size);
    if (world_rank == HEAD){ printf("Starting MPI_timing\n");}

    int solvetype = atoi(argv[1]);
    int tensortype = atoi(argv[2]);

    int* dvec = NULL;
    int d = 0;

    int* midvec = NULL;
    int mid = -1;

    int* heap_profile_vec = NULL;
    int heap_profile = 0;

    int* nps = NULL;

    int* n = NULL;
    int* r = NULL;

    void (*f_ten)(double* restrict, int*, int*, const void*);
    void* parameters = NULL;

    long* total_allocs = NULL;

    double acc = 1e-10;



    // Taken chars: 'd', 'm', 'n', 'r', 'p', 'e', 'g'

    dvec = get_int_star_input('d', 1, argc, argv);
    if (dvec == NULL){
        printf("No input d for Hilbert Tensor\n");
        MPI_Finalize();
        return 0;
    }
    else{
        d = dvec[0];
    }

    midvec = get_int_star_input('m', 1, argc, argv);
    if (midvec != NULL){
        mid = midvec[0];
    }

    // 'e' is for 'eeeeeeeeh-nps'
    nps = get_int_star_input('e', d, argc, argv);
    if (nps == NULL){
        printf("No input nps\n");
        MPI_Finalize();
        return 0;
    }

    n = get_int_star_input('n', d, argc, argv);
    if (n == NULL){
        printf("No input n\n");
        MPI_Finalize();
        return 0;
    }

    r = get_int_star_input('r', d+1, argc, argv);
    if (r == NULL){
        printf("No input r\n");
        MPI_Finalize();
        return 0;
    }

    char* path = get_string_input ('p', argc, argv);
    if (path == NULL){
        printf("Must input path with -p\n");
        MPI_Finalize();
        return 0;
    }

    heap_profile_vec = get_int_star_input('h', 1, argc, argv);
    if (heap_profile_vec != NULL){
        heap_profile = *heap_profile_vec;
        total_allocs = (long*) calloc(3, sizeof(long));
    }


    // Get the tensor pointer (I have _no_ idea how to make a function for this anymore)
    if (tensortype == 1){
        f_ten = &f_hilbert;
        parameters = p_hilbert_init(d, n);
    }
    else if (tensortype == 2){
        int* Mvec = NULL;
        int M;
        Mvec = get_int_star_input('M', 1, argc, argv);
        if (Mvec != NULL){
            M = Mvec[0];
        }
        else{
            printf("Must input M with -M when tensortype == 2\n");
            MPI_Finalize();
            return 0;
        }

        double* gammavec = NULL;
        double gamma;
        gammavec = get_double_star_input('g', 1, argc, argv);
        if (gammavec != NULL){
            gamma = gammavec[0];
        }
        else{
            printf("Must input gamma with -g when tensortype == 2\n");
            MPI_Finalize();
            return 0;
        }

        int seed = 168234591; // Arbitrary number
        parameters = unit_random_p_gaussian_bumps_init(d, n, M, gamma, seed);
        f_ten = &f_gaussian_bumps;

        free(gammavec); gammavec = NULL;
        free(Mvec);     Mvec = NULL;
    }

    MyHeapProfilerStart(heap_profile, path, ".tt", world_rank);
    tensor_train* tt = TT_init_rank(d, n, r);
    MyHeapProfilerStop(&heap_profile, path, ".tt", world_rank, total_allocs);

    MyHeapProfilerStart(heap_profile, path, ".ten", world_rank);
    MPI_tensor* ten = MPI_tensor_init(d, n, nps, MPI_COMM_WORLD, f_ten, parameters);
    MyHeapProfilerStop(&heap_profile, path, ".ten", world_rank, total_allocs + 1);

    VTime_gathered* tms = get_TT(tt, ten, mid, solvetype, &heap_profile, path, total_allocs + 2);
    if (world_rank == HEAD){ printf("Got the tensor train\n");}

    if (solvetype == 1){
        ten = MPI_tensor_init(d, n, nps, MPI_COMM_WORLD, f_ten, parameters);
    }
    double err = tt_error(tt, ten);
    if (world_rank == HEAD){
        printf("ERROR = %e\n", err);
        printf("TIME = %f\n", tms->t[0]);
    }
    MPI_tensor_free(ten);

    long* total_allocs_matrix = gather_heap_data(heap_profile, total_allocs, world_size, world_rank, comm_world);
    if (world_rank == HEAD){
        my_tt_write(tt, tms, err, nps, tensortype, solvetype, path, total_allocs_matrix);
    }


    if (tms != NULL){;
        VTime_gathered_free(tms); tms = NULL;
    }
    TT_free(tt); tt = NULL;

    if (tensortype == 1){
        p_hilbert_free(parameters); parameters = NULL;
    }
    else if (tensortype == 2){
        p_gaussian_bumps_free(parameters); parameters = NULL;
    }

    free(r);      r = NULL;
    free(n);      n = NULL;
    free(nps);    nps = NULL;
    free(midvec); midvec = NULL;
    free(dvec);   dvec = NULL;
    free(total_allocs); total_allocs = NULL;
    free(heap_profile_vec); heap_profile_vec = NULL;

    MPI_Finalize();

    return 0;
}

/* Key for argv[1]
 *  1 : streaming_SSTT
 *  2 : streaming_PSTT2
 *  3 : streaming_PSTT2_onepass
*/
VTime_gathered* get_TT(tensor_train* tt, MPI_tensor* ten, int mid, int solvetype, int *heap_profile, char* path, long* total_alloc_ptr)
{
    VTime* tm;
    int world_rank = ten->rank;
    MPI_Comm comm = ten->comm;

    if (world_rank==HEAD){ printf("Getting the tensor train ("); }
    if (solvetype == 1){
        if (world_rank==HEAD){ printf("SSTT)\n"); }
        MyHeapProfilerStart(*heap_profile, path, ".SSTT", world_rank);
        tm = SSTT(tt, ten);
        MyHeapProfilerStop(heap_profile, path, ".SSTT", world_rank, total_alloc_ptr);
    }
    if (solvetype == 2){
        if (world_rank==HEAD){ printf("PSTT2)\n"); }
        MyHeapProfilerStart(*heap_profile, path, ".PSTT", world_rank);
        tm = PSTT2(tt, ten, mid);
        MyHeapProfilerStop(heap_profile, path, ".PSTT", world_rank, total_alloc_ptr);
    }
    if (solvetype == 3){
        if (world_rank==HEAD){ printf("PSTT2_onepass)\n"); }
        MyHeapProfilerStart(*heap_profile, path, ".PSTT_onepass", world_rank);
        tm = PSTT2_onepass(tt, ten, mid);
        MyHeapProfilerStop(heap_profile, path, ".PSTT_onepass", world_rank, total_alloc_ptr);
    }
    if (world_rank==HEAD){ printf("Got tensor train\n");}

    VTime_gathered* tms = VTime_gather(tm, comm);

    if (tm != NULL){
        VTime_free(tm); tm = NULL;
    }

    return tms;
}