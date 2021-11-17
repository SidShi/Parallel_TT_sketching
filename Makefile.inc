#########################################################
# paralleltt example include file
#########################################################

# Archiver information
AR=ar
ARFLAGS=cr

# C compiler and flags
CC=mpicc

# Optimization flags
OPTFLAGS=-O3 -march=native -ffast-math -g
CFLAGS+=$(OPTFLAGS)

# Required libraries
LIBLAPACK=-llapacke -llapack
LIBBLAS=-lblas

LIBS=-ltcmalloc -lm -lprofiler $(LIBLAPACK) $(LIBBLAS)

