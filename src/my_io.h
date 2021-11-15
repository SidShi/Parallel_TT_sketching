
#ifndef FILE_BUF
#define FILE_BUF ((int) 256)
#endif

#ifndef MY_IO_H
#define MY_IO_H

#include "tt.h"
#include "VTime.h"
#include <gperftools/heap-profiler.h>

// Get a string input
char* get_string_input (const char c, int argc, char *argv[]);

// Get an input integer array
int* get_int_star_input(const char c, const int len, int argc, char *argv[]);

// Get an input double array
double* get_double_star_input(const char c, const int len, int argc, char *argv[]);

// Check if a directory exists
int isDirectoryExists(const char *path);

// Append nextFolder onto path
void mycd(char *path, const char *nextFolder);

// Get the input path (seems useless)
//char* get_path(const tensor_train* tt, const char* input_path, int tensortype, int solvetype);

// Save a matrix of integers
void save_int_matrix(int m, int n, const int* A, const char* path, const char* file);

// Save a matrix of longs
void save_long_matrix(int m, int n, const long* A, const char* path, const char* file);

// Save a matrix of doubles
void save_double_matrix(int m, int n, const double* A, const char* path, const char* file);

// Save a matrix of strings
void save_string_matrix(int m, int n, char** A, const char* path, const char* file);

// NOTE: tt_write might be some library function or something...
void my_tt_write(const tensor_train* tt, VTime_gathered* tms, const double err, const int* nps, int tensortype,
                 int solvetype, char* path, const long* total_allocs_matrix);

#endif


