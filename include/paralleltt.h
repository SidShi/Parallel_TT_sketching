/* paralleltt.h
 * This is the main header file for the parallel tensor train decomposition package. Documentation for specific
 * functions, such as PSTT2, PSTT2_onepass, and SSTT can be found in the individual header files in ../src/
 */

// Implementation of a matrix datatype and the associated routines
#include "./matrix.h"

// The tensor datatype and associated functiosn
#include "./tensor.h"

// The tensor-train datatype and associated functions
#include "./tt.h"

// The sketch datatype
#include "./sketch.h"

// The implementation of SSTT
#include "./SSTT.h"

// THe implementation of PSTT2 and PSTT2-onepass
#include "./PSTT.h"
