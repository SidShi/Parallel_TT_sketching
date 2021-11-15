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
- `void (*f_ten)(double* X, int* ind1, int* ind2, const void* parameters)` is a function that loads entries of the sub-tensor. The location of the subtensor in the tensor is determined by `ind1` and `ind2`, where `ind1` is dimension `d` array giving the lower bounds of a 

# Required Packages
The user of this code must provide implementations of MPI, [LAPACKE](https://www.netlib.org/lapack/lapacke.html), and BLAS.


[1] **Reference for submission**  
[2] I. V. Oseledets, _Tensor-train decomposition_, SIAM J. Sci. Comput., 33 (2011), pp. 2295-2317  
[3] M. Che and Y. Wei, _Randomized algorithms for the approximations of Tucker and the tensor train decompositions_, Adv. Comput. Math., 45 (2019), pp. 395-428
