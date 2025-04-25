#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>

#include "common.h"
#include "matrix.hpp"

// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}

int find_int_arg(int argc, char** argv, const char* option, int default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return std::stoi(argv[iplace + 1]);
    }

    return default_value;
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv); // Initialize the MPI environment
  
  int n_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &n_ranks); // Get the number of processes
  
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); // Get the rank of the process

  if (find_arg_idx(argc, argv, "-h") >= 0) {
      std::cout << "-N <int>: side length of the sparse matrix" << std::endl;
      return 0;
  }

  // int N = find_int_arg(argc, argv, "-N", 1 << 20); // global size
  int N = 10;

  assert(N % n_ranks == 0);
  int n = N / n_ranks; // number of local rows


  // Create local matrix
  CSR A_local(n, N);
  fill_local_matrix(A_local, my_rank);

  // Print matrix
  for (int i=0; i < n_ranks; i++) {

	  if (my_rank == i) {
		  std::cout << "Rank " << my_rank << ":" << std::endl;

		  std::cout << Eigen::MatrixXd(A_local) << std::endl;
	  }

	  MPI_Barrier(MPI_COMM_WORLD);
  }




  MPI_Finalize(); // Finalize the MPI environment

  return 0;
}
