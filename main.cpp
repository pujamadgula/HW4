#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>
#include <thread>
#include <chrono>

#include <mpi.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#
#include "common.h"

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


typedef Eigen::SparseMatrix<double, Eigen::RowMajor>     CSR;
typedef Eigen::Triplet<double>                           Triplet;
typedef Eigen::VectorXd                                  Vec;
typedef Eigen::VectorBlock<Eigen::VectorXd>              VecView;


// making the 1D laplacian for a local chunk of the 
// matrix on a specific rank
void fill_local_matrix(CSR& A_local, int my_rank) {

	int n = A_local.rows();
	int N = A_local.cols();
	std::vector<Triplet> triplets;

	int row_start = my_rank * n;
	for (int i=0; i < n; ++i) {
		int gi = row_start + i;

		if (gi > 0) triplets.emplace_back(i, gi-1, -1.0);

		triplets.emplace_back(i, gi, 2.0);

		if (gi < N-1) triplets.emplace_back(i, gi+1, -1.0);

	}

	A_local.setFromTriplets(triplets.begin(), triplets.end());
	A_local.makeCompressed();
}


// Extract the square block of elements on A_local which 
// fall along the diagonal of the whole matrix A
void get_diagonal_block(const CSR& A_local, CSR& A_block, int my_rank) {

	int n = A_local.rows();
	int row_start = my_rank * n;

        std::vector<Triplet> triplets;

	for (int i=0; i < n; ++i) {
		for (CSR::InnerIterator it(A_local, i); it; ++it) {
			int col = it.col();

			if (col >= row_start && col < row_start + n) {
				int local_col = col - row_start;
				triplets.emplace_back(i, local_col, it.value());
			}
		}
	}

	A_block.setFromTriplets(triplets.begin(), triplets.end());
	A_block.makeCompressed();
}


// 
void apply_preconditioner(const CSR& A_block, const Vec& r, Vec& r_cond, int my_rank) {

	// Only need to create once
	// fresh each time for debugging
	static Eigen::IncompleteCholesky<double, Eigen::Lower, Eigen::NaturalOrdering<int>> ichol;
	static bool initalized = false;

	if (!initalized) {
	    std::cout << "Rank " << my_rank << " initalizing preconditioner" << std::endl;
   	    ichol.compute(A_block);
	    if (ichol.info() != Eigen::Success) {
                throw std::runtime_error("CHOL INIT FAILED!");
	    }
	    initalized = true;
	}

	r_cond = ichol.solve(r);

	if (ichol.info() != Eigen::Success) {
		throw std::runtime_error("PRECONDITIONER SOLVE FIAILED");
	}

	//std::cout << "Rank " << my_rank << ": r.head(5) = " << r.head(5).transpose() << "\n";
        //std::cout << "Rank " << my_rank << ": r_cond.head(5) = " << r_cond.head(5).transpose() << "\n";
}

/*
void exchange_P_halo(int my_rank, int n_ranks, int n, Vec& P_halo) {

	MPI_Request requests[4];
	int req_idx = 0;

	int row_start = my_rank * n;
	int row_end   = my_rank * (n+1);

	double for_left, for_right;
	double from_left, from_right;

	// R0  - R1 - R2 - R3 - Rn_ranks-1 //

	// Send to left
	if (my_rank > 0) {
		for_left = P_halo[1];
		MPI_Isend(&for_left, 1, MPI_DOUBLE, my_rank-1, 0, MPI_COMM_WORLD, &requests[req_idx++]);
	}

	// Send to right
	if (my_rank < n_ranks -1) {
		for_right = P_halo[1 + n];
		MPI_Isend(&for_right, 1, MPI_DOUBLE, my_rank+1, 0, MPI_COMM_WORLD, &requests[req_idx++]);
	}

	// Recieve from left
	if (my_rank > 0) {
		MPI_Irecv(&from_left, 1, MPI_DOUBLE, my_rank-1, 0, MPI_COMM_WORLD, &requests[req_idx++]);
	}

	// Recieve from right
	if (my_rank < n_ranks-1) {
		MPI_Irecv(&from_right, 1, MPI_DOUBLE, my_rank+1, 0, MPI_COMM_WORLD, &requests[req_idx++]);
	}

	MPI_Waitall(req_idx, requests, MPI_STATUSES_IGNORE);

	// Set exchanged values in P
	if (my_rank > 0) {
		P_halo[0] = from_left;
	}

	if (my_rank < n_ranks-1) {
		P_halo[1 + n] = from_right;
	}
}
*/

void exchange_P_halo(int my_rank, int n_ranks, int n, Vec& P_halo) {
    // P_halo layout: [ 0 | 1..n | n+1 ]
    //                ^ghost   real   ^ghost

    MPI_Request reqs[4];
    int cnt = 0;

    // 1) non‑blocking send of first real to left‑neighbour’s right‑ghost
    if (my_rank > 0) {
        MPI_Isend(&P_halo[1],      1, MPI_DOUBLE, my_rank-1, 0, MPI_COMM_WORLD, &reqs[cnt++]);
        MPI_Irecv(&P_halo[0],      1, MPI_DOUBLE, my_rank-1, 0, MPI_COMM_WORLD, &reqs[cnt++]);
    }

    // 2) non‑blocking send of last real to right‑neighbour’s left‑ghost
    if (my_rank < n_ranks-1) {
        MPI_Isend(&P_halo[n],      1, MPI_DOUBLE, my_rank+1, 0, MPI_COMM_WORLD, &reqs[cnt++]);
        MPI_Irecv(&P_halo[n+1],    1, MPI_DOUBLE, my_rank+1, 0, MPI_COMM_WORLD, &reqs[cnt++]);
    }

    MPI_Waitall(cnt, reqs, MPI_STATUSES_IGNORE);

    // debug print
    std::cout << "Rank " << my_rank << " after halo exchange, P_halo = [";
    for (int i = 0; i < n+2; ++i) std::cout << P_halo[i] << (i<n+1?", ":"");
    std::cout << "]\n";
}


int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int n_ranks, my_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  int N = 100;
  assert(N % n_ranks == 0);
  int n = N / n_ranks;
  int row_start = my_rank * n;
  int row_end = (my_rank+1) * n;

  std::cout << "Rank " << my_rank << ": owns rows [" << row_start << ", " << row_end << ")\n";


  // build chunk of A
  CSR A_local(n, N);
  fill_local_matrix(A_local, my_rank);
  CSR A_block(n,n);
  get_diagonal_block(A_local, A_block, my_rank);

  std::cout << "Rank " << my_rank << " passed this section" << std::endl;

  // Initialize vectors
  Vec x = Vec::Zero(n), 
      b = Vec::Ones(n),
      r = Vec::Ones(n),
      r_cond(n), 
      AP(n);

  const double epsilon =  1e-8 * std::sqrt(b.dot(b));


  // initial preconditioning
  apply_preconditioner(A_block, r, r_cond, my_rank);

  double prev_rr_local = r.dot(r_cond);
  double prev_rr;
  MPI_Allreduce(&prev_rr_local, &prev_rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


  // build P_halo and view of P subset
  Vec P_halo = Vec::Zero(n+2);
  P_halo.segment(1, n) = r_cond;
  VecView P = P_halo.segment(1, n);




  int iter = 0;
  double res_norm;
  while (iter < 10) {

    // exchange halo
    exchange_P_halo(my_rank, n_ranks, n, P_halo);

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i=0; i < n_ranks; i++) {
	    if (i == my_rank) {
                std::cout << "Rank " << my_rank << ": P_halo = [";
                for (int i = 0; i < n + 2; ++i)
                    std::cout << P_halo[i] << (i < n + 1 ? ", " : "");
                std::cout << "]\n";
	    }
	    MPI_Barrier(MPI_COMM_WORLD);
    }

    // SpMV
    for (int local_row = 0; local_row < n; ++local_row) {
      AP[local_row] = 0;

      for (CSR::InnerIterator it(A_local, local_row); it; ++it) {
        int global_col = it.col();

	int P_idx = global_col - row_start + 1;

        if (P_idx < 0 || P_idx >= n + 2) {
            std::cerr << "Rank " << my_rank << " ERROR: P_idx out of bounds (" << P_idx
              << ") at row " << local_row << ", col " << global_col << ", row_start " << row_start << "\n";
         }

        AP[local_row] += it.value() * P_halo[P_idx];
      }
    }

    // alpha
    double local_pap = P.dot(AP);
    double global_pap;
    MPI_Allreduce(&local_pap, &global_pap, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double alpha = prev_rr / global_pap;                   

    x += alpha * P;                                          
    r -= alpha * AP;

    // precondition new residual
    apply_preconditioner(A_block, r, r_cond, my_rank);
    double new_rr_local = r.dot(r_cond);
    double new_rr;
    MPI_Allreduce(&new_rr_local, &new_rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // bail early if meets tolerance
    res_norm = std::sqrt(new_rr);
    if (res_norm < epsilon)
        break;

    // beta
    double beta = new_rr / prev_rr;                     
    prev_rr = new_rr;

    // update search path
    Vec P_old = P; // Copies current values of P
    P = r_cond + beta * P_old;

    ++iter;
  }

  if (my_rank == 0)
      std::cout << "iter=" << iter
                << "  res_norm=" << res_norm
		<< "  eps=" << epsilon
		<< std::endl;


  Vec x_global;
  if(my_rank==0) x_global.resize(N);

  MPI_Gather(x.data(), n, MPI_DOUBLE, my_rank==0 ? x_global.data() : nullptr, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if(my_rank==0){
    std::cout << "gathered x = [ ";
    for(int i=0; i<N; ++i) std::cout << x_global[i] << " ";
    std::cout << "]\n\n";
  }

  if (my_rank == 0) {
    double r_square = 0;
    for (int i = 0; i < N; ++i) {
      double r = x_global[i] * 2;
      if (i > 0)  r -= x_global[i - 1];
      if (i + 1 < N)  r -= x_global[i + 1];
      r_square += (r - 1) * (r - 1);
    }
    std::cout << "|Ax - b| / |b| = " << std::sqrt(r_square) / std::sqrt(N) << std::endl;
  }


  MPI_Finalize();
  return 0;
}

