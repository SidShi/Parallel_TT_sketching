// Who knows what the V stands for anymore. This is just the timing wrapper that I will use to time things.

#ifndef VTIME_H
#define VTIME_H

#include <gperftools/heap-profiler.h>

typedef struct VTime
{
    double* t;      // A list of times stored by the VTime
    char** labels;  // A list of labels for each of the times
    int n_t;        // The number of times stored
    int offset;     // Offset of times, used for passing the object into functions so they can start at zero

    double t_init;  // Time at initialization
    double t_break; // Time of the most recent break
    double t_tmp;   // Used for temporary when breaking
    double t_final; // Time when finalized
} VTime;

VTime* VTime_init(int n_t);                            // Initialize a VTime object
void VTime_set_label(VTime* tm, int ind, char* label); // Set the label of a given index
void VTime_break(VTime* tm, int ind, char* label);     // Get the time since the last break. If ind = -1, do not save it
void VTime_finalize(VTime* tm);                        // Finalize a VTime object, write the total time to t[0]
void VTime_free(VTime* tm);                            // Free a VTime
void VTime_print(VTime* tm);                           // Print a VTime

typedef struct VTime_gathered
{
    double* t;     // A matrix of times from every core
    char** labels; // Labels for the categories, assumed to be the same for every core
    int n_t;       // The number of times stored per core
    int size;      // The world size
} VTime_gathered;

VTime_gathered* VTime_gather(VTime* tm, MPI_Comm comm);
void VTime_gathered_free(VTime_gathered* tms);

#endif