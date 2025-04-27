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


  // Set up for communication
  std::vector<int> ghost_cols;
  std::unordered_map<int, int> g2l;
  std::vector<std::vector<int>> send_ranks, recv_ranks;
  compute_halo(A_local, my_rank, n_ranks, N, ghost_cols, send_ranks, recv_ranks, g2l);


  MPI_Barrier(MPI_COMM_WORLD);  
  for (int r = 0; r < n_ranks; ++r) {
      if (r == my_rank) {
          std::cout << "==== Rank " << my_rank << " ====" << std::endl;
  
          // Print ghost_cols
          std::cout << "ghost_cols: ";
          for (int g : ghost_cols)
              std::cout << g << " ";
          std::cout << std::endl;
  
          // Print g2l map
          std::cout << "g2l mapping: ";
          for (const auto &pair : g2l)
              std::cout << "[" << pair.first << " -> " << pair.second << "] ";
          std::cout << std::endl;
  
          // Print send_ranks
          std::cout << "send_ranks:" << std::endl;
          for (int i = 0; i < n_ranks; ++i) {
              if (!send_ranks[i].empty()) {
                  std::cout << "  to rank " << i << ": ";
                  for (int idx : send_ranks[i])
                      std::cout << idx << " ";
                  std::cout << std::endl;
              }
          }
  
          // Print recv_ranks
          std::cout << "recv_ranks:" << std::endl;
          for (int i = 0; i < n_ranks; ++i) {
              if (!recv_ranks[i].empty()) {
                  std::cout << "  from rank " << i << ": ";
                  for (int idx : recv_ranks[i])
                      std::cout << idx << " ";
                  std::cout << std::endl;
              }
          }
  
          std::cout << "===================" << std::endl << std::endl;
          std::cout.flush();
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      MPI_Barrier(MPI_COMM_WORLD);  // synchronize after each my_rank prints
  }



  // start setting up for CG
  // solution vector
  Vec x_local = Vec::Zero(n);

  Vec r_local(n); // residual (think gradient)
		  
  Vec search_path(n);   // Updated at each step base on that step's residue
			     // combines with previous

 
  Vec r_local_cond(n); // residual after preconditioning applied
  Vec prev_r_local_cond(n); // prevoius step's conditioned residual 


  Vec b_local = Vec::Ones(n); // ax - b = 0
  

  // At the beginning, the residual IS b since
  // our fist guess is all 0s
  r_local = b_local;

  // Apply preconditioner
  apply_preconditioner(A_block, r_local, r_local_cond);

  // set the first step direction
  search_path = r_local_cond;


  // error
  double r_dot_local = r_local.dot(r_local_cond);
  double r_dot_global;
  MPI_Allreduce(&r_dot_local, &r_dot_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


  if (my_rank == 0) {
      std::cout << "starting error: " << r_dot_global << std::endl;
  }


  Vec result; // product of A_local and search_path


  // GETTING A DEADLOCK WHEN CALLING THE EXCHANGE FUNC

  /*
  int max_iters = 100;

  for (int k=0; k < max_iters; ++k) {
	  Vec p_ghost;
	  exchange(my_rank, n_ranks, search_path, ghost_cols, send_ranks, recv_ranks, p_ghost);

	  local_m_v(A_local, search_path, p_ghost, g2l, result);

	  double pAp_local = search_path.dot(result);
	  double pAp_global;
	  MPI_Allreduce(&pAp_local, &pAp_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	  // step size
	  double alpha = r_dot_global / pAp_global;

	  x_local += alpha * search_path;
	  r_local -= alpha * result;

	  apply_preconditioner(A_block, r_local, prev_r_local_cond);

	  double r_dot_new_local = r_local.dot(prev_r_local_cond);
	  double r_dot_new_global;
	  MPI_Allreduce(&r_dot_new_local, &r_dot_new_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	  // check if we meet tolerance
	  if (std::sqrt(r_dot_new_global) < 1e-6) break;


	  // If not passing, update path
	  double beta = r_dot_new_global / r_dot_global;
	  search_path = prev_r_local_cond + beta * search_path;
	  r_dot_global = r_dot_new_global;

	  std::cout << "finished iter " << k << std::endl;

  }

  if (my_rank == 0) std::cout << "CG finished" << std::endl;

  */
  

  MPI_Finalize(); // Finalize the MPI environment

  return 0;
}
