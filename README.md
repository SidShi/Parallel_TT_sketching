# Parallel Streaming Tensor Train Sketching
This repository contains C implementations for three algorithms described in [1] for the parallel computatation of the tensor train (TT) decomposition [2]. The algorithms are
- _Two-sided Parallel Streaming TT Sketching (PSTT2):_ a low-memory method for computation of tensor trains.
- _One-pass Two-sided Parallel Streaming TT Sketching (PSTT2-onepass):_ a single-pass method for performing PSTT2.
- _Serial Streaming TT Sketching (SSTT):_ a parallel implementation of algorithm 5.1 in [3].  

# How to use
To run the code, first one allocates an `MPI_tensor` object. The `MPI_tensor` contains all of the necessary information to stream "sub-tensors" for the purposes of the above algorithms (see [1] for the definition of sub-tensors). One initializes an `MPI_tensor` with the command  
> `MPI_tensor* MPI_tensor_init(int d, const int* n, int* nps, MPI_Comm comm, void (*f_ten)(double* restrict, int*, int*, const void*), void* parameters)`  

where 
- `int d` is the dimension of the tensor.
- `int* n` are the sizes in each dimension.
- `int* nps` designates the number of partitions in each dimension (the equivalent of **P** in [1]).
- `void (*f_ten)(double* X_sub, int* ind1, int* ind2, const void* parameters)` is a function that streams entries of the sub-tensor `X_sub`. The location of the sub-tensor in the tensor is determined by `ind1` and `ind2`. Conceptually, using MATLAB-like notation, this function streams `X_sub = X[ind1[0]:ind2[0]-1, ind1[1]:ind2[1]-1, ..., ind1[d-1]:ind2[d-1]-1]` in column-major order where `X` would be the full tensor.
- `void* parameters` is a struct used by `f_ten` to determine the entries, if necessary. 

One also initializes a tensor-train object with
> `tensor_train* TT_init_rank(const int d, const int* restrict n, const int* restrict r)`

where `r` is a length `d+1` vector with the ranks that we want to calculate the TT with. Once these two data structures are initialized, the three algorithms can be run with
> `void PSTT2(tensor_train* tt, MPI_tensor* ten, int mid)`
> `void PSTT2_onepass(tensor_train* tt, MPI_tensor* ten, int mid)`
> `void SSTT(tensor_train* tt, MPI_tensor* ten)`
where `mid` is the index of which TT core should be considered the "middle" core (use `mid=-1` to have the program calculate the index automatically). Each algorithm fills the TT `tt` stored by the 0th rank with the TT decomposition.

# Example
There is an example script included in `test>example.c`. To run the script, simply use the command
> `make test THREADS=nthreads`

One can edit `example.c` to try three different tensors with user-defined values of `d`, `n`, `r`, and `nps`. Two of these tensors are the Hilbert and Gaussian bumps tensors, defined in [1]. The last is the "arithmetic" tensor, with `X[i] = i+1`. 


# Required Packages
The user of this code must provide implementations of MPI, [LAPACKE](https://www.netlib.org/lapack/lapacke.html), and BLAS. These packages can be included in `Makefile.inc`. 

# References
[1] T. Shi, M. Ruth, and A. Townsend, _Parallel algorithms for computing the tensor-train decomposition_, submitted to SIAM J. Sci. Comput..  
[2] I. V. Oseledets, _Tensor-train decomposition_, SIAM J. Sci. Comput., 33 (2011), pp. 2295-2317  
[3] M. Che and Y. Wei, _Randomized algorithms for the approximations of Tucker and the tensor train decompositions_, Adv. Comput. Math., 45 (2019), pp. 395-428
