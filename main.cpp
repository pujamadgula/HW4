#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>
#include <thread>
#include <chrono>

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

  CSR A_block(n,n);
  get_diagonal_block(A_local, A_block, my_rank);

  // Print matrix
  MPI_Barrier(MPI_COMM_WORLD);  
  for (int i=0; i < n_ranks; i++) {

	  if (my_rank == i) {
		  std::cout << "Rank " << my_rank << " chunk:" << std::endl;

		  std::cout << Eigen::MatrixXd(A_local) << std::endl;
		  std::cout << std::endl;
		  std::cout.flush();
		  std::this_thread::sleep_for(std::chrono::milliseconds(100));
	  }

	  MPI_Barrier(MPI_COMM_WORLD);
  }

  // Print diagonal blocks
  MPI_Barrier(MPI_COMM_WORLD);  
  for (int i=0; i < n_ranks; i++) {

	  if (my_rank == i) {
		  std::cout << "Rank " << my_rank << " diag block:" << std::endl;
		  std::cout << Eigen::MatrixXd(A_block) << std::endl;
		  std::cout << std::endl;
		  std::cout.flush();
		  std::this_thread::sleep_for(std::chrono::milliseconds(100));
	  }

	  MPI_Barrier(MPI_COMM_WORLD);
  }


  Vec B = Vec::Ones(n); // ax - b = 0
  
  // At the beginning, the residual IS b since
  // our fist guess is all 0s
  Vec x = Vec::Zero(n);
  Vec r = Vec::Ones(n);

  // Apply preconditioner
  Vec r_cond = Vec::Zero(n);
  apply_preconditioner(A_block, r, r_cond);

  double prev_rr = r.dot(r_cond);

  // set the first step direction
  // as the pre-conditioned residual

  Vec P_halo;
  if (my_rank == 0 || my_rank == n_ranks-1) {
	  P_halo = Vec::Zero(n+1);
  } else {
	  P_halo = Vec::Zero(n+2);
  }

  int local_start;
  if (my_rank == 0) {
	  local_start = 0;
  } else {
	  local_start = 1;
  }

  P_halo.segment(local_start, n) = r_cond;
  VecView P = P_halo.segment(local_start, n);

  std::cout << "initialization done" << std::endl;

  int iter = 0;
  while(iter < 1000) {

	  std::cout << "entering iteration " << iter << std::endl;

	  // Need to populate P with values
	  // from other ranks
	  exchange_P_halo(my_rank, n_ranks, n, local_start, P_halo);

	  std::cout << "Got P_halo" << std::endl;

	  // Multiply
	  Vec AP(n);

	  std::cout << "finished multiply" << std::endl;

	  // alpha
	  double pAP = P.dot(AP);
	  double local_alpha = prev_rr / pAP;
	  double global_alpha;
	  MPI_Allreduce(&local_alpha, &global_alpha, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


	  std::cout << "all reduce alpha" << std::endl;

	  // Take step
	  x += (global_alpha * P);

	  // New residual
	  r -= (global_alpha * AP);

	  // Precondition our new residual
          apply_preconditioner(A_block, r, r_cond);

	  // beta
	  double new_rr = r.dot(r_cond);
	  double local_beta = new_rr / prev_rr;
	  double global_beta;
	  MPI_Allreduce(&local_beta, &global_beta, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


	  // Update search path
	  P = r_cond + (global_beta * P);

	  // Set up next iteration
	  prev_rr = new_rr;
          iter++;
  }


  MPI_Finalize(); // Finalize the MPI environment

  return 0;
}
