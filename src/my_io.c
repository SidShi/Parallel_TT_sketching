#include "my_io.h"
#include "tt.h"

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <gperftools/heap-profiler.h>

#ifndef FILE_BUF
#define FILE_BUF ((int) 256)
#endif


// Input:
//   c - character flag that we are looking for,
//   argc, *argv[] - inputs to main
char* get_string_input (const char c, int argc, char *argv[])
{
    char* str = NULL;

    for (int ii = 1; ii < argc; ++ii)
    {
        char* arg = argv[ii];
        if (arg[0] == '-'){
            if (arg[1] == c){// Output path specified
                str = argv[ii+1];
            }
        }
    }

    return str;
}

// Input:
//   c - character flag that we are looking for,
//   len - length of int array we are retrieving
//   argc, *argv[] - inputs to main
int* get_int_star_input(const char c, const int len, int argc, char *argv[])
{
    int* vec = (int*) malloc(sizeof(int)*len);
    int specified = 0;

    for (int ii = 1; ii < argc; ++ii)
    {
        char* arg = argv[ii];
        if (arg[0] == '-'){
            if (arg[1] == c){// Output path specified
                specified = 1;
                for (int jj = 0; jj < len; ++jj){
                    vec[jj] = atoi(argv[ii+jj+1]);
                }
            }
        }
    }

    if (specified == 0){
        free(vec); vec = NULL;
        vec = NULL;
    }
    return vec;
}

double* get_double_star_input(const char c, const int len, int argc, char *argv[])
{
    double* vec = (double*) malloc(sizeof(double)*len);
    int specified = 0;

    for (int ii = 1; ii < argc; ++ii)
    {
        char* arg = argv[ii];
        if (arg[0] == '-'){
            if (arg[1] == c){// Output path specified
                specified = 1;
                for (int jj = 0; jj < len; ++jj){
                    vec[jj] = atof(argv[ii+jj+1]);
                }
            }
        }
    }

    if (specified == 0){
        free(vec); vec = NULL;
        vec = NULL;
    }
    return vec;
}

// Don't ask my how this works...
int isDirectoryExists(const char *path)
{
    struct stat file_stat;

    if (stat(path, &file_stat) < 0) {
        return 0;
    }

    return S_ISDIR(file_stat.st_mode);
}/**/


void mycd(char *path, const char *nextFolder)
{
    char tmp[FILE_BUF];

    int ret = snprintf(tmp, FILE_BUF, "%s%s", path, nextFolder);
    if (ret < 0) { abort(); }
    ret = snprintf(path, FILE_BUF, "%s", tmp);
    if (ret < 0) { abort(); }
    int dirExists = isDirectoryExists(path);
    if (!dirExists) { mkdir(path, 0777); }
}

//char* get_empty_path(const tensor_train* tt, const char* input_path, int tensortype, int solvetype)
//{
//    char* path = malloc(sizeof(char) * FILE_BUF);
//    path[0] = '\0';
//
//    mycd(path, input_path);
//
//
//    return path;
//}

void save_int_matrix(int m, int n, const int* A, const char* path, const char* file)
{
    char A_file[FILE_BUF];
    int ret = snprintf(A_file, FILE_BUF, "%s%s", path, file);
    if (ret < 0) { abort(); }

    FILE* stream = fopen(A_file,"w");
    for (int ii = 0; ii < m; ++ii){
        for (int jj = 0; jj < n; ++jj){
            if (jj == 0){
                fprintf(stream, "%d", A[jj*m + ii]);
            }
            else{
                fprintf(stream, ",%d", A[jj*m +ii]);
            }
        }
        if (ii != m-1){
            fprintf(stream, "\n");
        }
    }
    fclose(stream);
}


void save_long_matrix(int m, int n, const long* A, const char* path, const char* file)
{
    char A_file[FILE_BUF];
    int ret = snprintf(A_file, FILE_BUF, "%s%s", path, file);
    if (ret < 0) { abort(); }

    FILE* stream = fopen(A_file,"w");
    for (int ii = 0; ii < m; ++ii){
        for (int jj = 0; jj < n; ++jj){
            if (jj == 0){
                fprintf(stream, "%ld", A[jj*m + ii]);
            }
            else{
                fprintf(stream, ",%ld", A[jj*m +ii]);
            }
        }
        if (ii != m-1){
            fprintf(stream, "\n");
        }
    }
    fclose(stream);
}

void save_double_matrix(int m, int n, const double* A, const char* path, const char* file)
{
    char A_file[FILE_BUF];
    int ret = snprintf(A_file, FILE_BUF, "%s%s", path, file);
    if (ret < 0) { abort(); }

    FILE* stream = fopen(A_file,"w");
    for (int ii = 0; ii < m; ++ii){
        for (int jj = 0; jj < n; ++jj){
            if (jj == 0){
                fprintf(stream, "%e", A[jj*m + ii]);
            }
            else{
                fprintf(stream, ",%e", A[jj*m + ii]);
            }
        }
        if (ii != m-1){
            fprintf(stream, "\n");
        }
    }
    fclose(stream);
}

void save_string_matrix(int m, int n, char** A, const char* path, const char* file)
{
    char A_file[FILE_BUF];
    int ret = snprintf(A_file, FILE_BUF, "%s%s", path, file);
    if (ret < 0) { abort(); }

    FILE* stream = fopen(A_file,"w");
    for (int ii = 0; ii < m; ++ii){
        for (int jj = 0; jj < n; ++jj){
            if (jj == 0){
                fprintf(stream, "%s", A[jj*m + ii]);
            }
            else{
                fprintf(stream, ",%s", A[jj*m + ii]);
            }
        }
        if (ii != m-1){
            fprintf(stream, "\n");
        }
    }
    fclose(stream);
}

void my_tt_write(const tensor_train* tt, VTime_gathered* tms, const double err, const int* nps, int tensortype,
                 int solvetype, char* path, const long* total_allocs_matrix)
{
    printf("Starting tt_write\n");

    int world_rank;
    int world_size;
    MPI_Comm comm_world = MPI_COMM_WORLD;
    MPI_Comm_rank(comm_world, &world_rank);
    MPI_Comm_size(comm_world, &world_size);

    int d = -1;
    double compression = -1.0;

    if(tt){
        d = tt->d;
        compression = get_compression(tt);
    }

    if (tt)        {save_int_matrix(1  , 1, &d   , path, "d.csv");}
    if (tt)        {save_int_matrix(d  , 1, tt->n, path, "n.csv");}
    if (tt)        {save_int_matrix(d+1, 1, tt->r, path, "r.csv");}
    if (tt && nps) {save_int_matrix(d  , 1, nps  , path, "nps.csv");}

    if (tms) {save_double_matrix(tms->n_t, tms->size, tms->t      , path, "t.csv");}
    if (err) {save_double_matrix(1       , 1        , &err        , path, "err.csv");}
    if (tt)  {save_double_matrix(1       , 1        , &compression, path, "compression.csv");}

    if (tms) {save_string_matrix(tms->n_t, 1, tms->labels, path, "t_labels.csv");}

    if (solvetype)  {save_int_matrix(1, 1, &solvetype  , path, "solvetype.csv");}
    if (tensortype) {save_int_matrix(1, 1, &tensortype , path, "tensortype.csv");}

    if (total_allocs_matrix) {save_long_matrix(3, 1 + world_size, total_allocs_matrix, path, "total_allocs.csv");}

    printf("path = %s\n", path);
}


