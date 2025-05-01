#pragma once

#include <mpi.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>


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
void apply_preconditioner(const CSR& A_block, const Vec& r, Vec& r_cond) {

	// Only need to create once
	static Eigen::IncompleteCholesky<double, Eigen::Lower, Eigen::NaturalOrdering<int>> ichol;
	static bool initialized = false;

	if (!initialized) {
		ichol.compute(A_block);
		if (ichol.info() != Eigen::Success) {
			throw std::runtime_error("CHOL INIT FAILED!");
		}
		initialized = true;
	}

	r_cond = ichol.solve(r);

	if (ichol.info() != Eigen::Success) {
		throw std::runtime_error("PRECONDITIONER SOLVE FIAILED");
	}
}


/// NEED TO VISUALIZE THE SPMV 
//
//   x0   x1   x2 |  x3   x4   x5
//    2   -1    0 |   0    0    0                | x0          | 
//   -1    2   -1 |   0    0    0    RANK 0      | x1          |
//    0   -1    2 | |-1|   0    0                | x2          |
//    ------------|--------------         x    --P-----   = result 
//    0    0  |-1||   2   -1    0                | x3          |
//    0    0    0 |  -1    2   -1    RANK 1      | x4          |
//    0    0    0 |   0   -1    2                | x5          |


// What's the 'rule' here
// if your column index is outside the range of your rows
// then you'll need to communicate
// but only a subset bc the rest are zeros
//
// should I just check if you are on the triangle???
// or is there a simplier way

//   x0   x1 | x2   x3  | x4   x5
//    2   -1 |  0    0  |  0    0                | x0          | 
//   -1    2 ||-1|   0  |  0    0   RANK 0       | x1          |
//    -------|----------|--------
//    0  |-1||  2   -1  |  0    0                | x2          |
//    0    0 | -2    2  ||-1|   0   RANK 1       | x3          |
//    -------|----------|--------
//    0    0 |  0  |-1| |  2   -1                | x4          |
//    0    0 |  0    0  | -1    2   Rank 2       | x5          |
//
//
//    wait does it even matter which thing is "outside" of
//    the bounds? bc I'm communicating p not A
//    so maybe that really doesnt matter
//    and I just need to know which neighbors to send my P to
//
//    I could probably get away with not sending the full P
//    if I did careful book keeping, but should I maybe start
//    with just sending all of P????
//


//  The communication pattern boils down to the same as a 1D neighbor
//  exchange


// what if i started with just a stupid simple version where
// i build all of the required P vector (padding with zeros feels stupid tho)...
//
// should we use a sparse vector maybe??????
// Let's see if we can set our functions up to be
// be somewhat flexible and decide later
//



// Take the current residual on this rank
// build the next P matrix
// by exchanging with neighbors
void exchange_P_halo(int my_rank, int n_ranks, int n, int local_start, Vec& P_halo) {

	MPI_Request requests[4];
	int req_idx = 0;

	int row_start = my_rank * n;
	int row_end   = my_rank * (n+1);

	double for_left, for_right;
	double from_left, from_right;

	// R0  - R1 - R2 - R3 - Rn_ranks-1 //

	// Send to left
	if (my_rank > 0) {
		for_left = P_halo[local_start];

		std::cout << my_rank << " sending first P value to " << my_rank-1 << std::endl;

		MPI_Isend(&for_left, 1, MPI_DOUBLE, my_rank-1, 0, MPI_COMM_WORLD, &requests[req_idx++]);
	}

	// Send to right
	if (my_rank < n_ranks -1) {
		for_right = P_halo[local_start + n];

		std::cout << my_rank << " sending last P value to " << my_rank+1 << std::endl;

		MPI_Isend(&for_right, 1, MPI_DOUBLE, my_rank+1, 0, MPI_COMM_WORLD, &requests[req_idx++]);
	}

	// Recieve from left
	if (my_rank > 0) {

		std::cout << my_rank << " recieving from " << my_rank-1 << std::endl;
		MPI_Irecv(&from_left, 1, MPI_DOUBLE, my_rank-1, 0, MPI_COMM_WORLD, &requests[req_idx++]);
	}

	// Recieve from right
	if (my_rank < n_ranks-1) {
		std::cout << my_rank << " recieving from " << my_rank+1 << std::endl;
		MPI_Irecv(&from_right, 1, MPI_DOUBLE, my_rank+1, 0, MPI_COMM_WORLD, &requests[req_idx++]);
	}

	MPI_Waitall(req_idx, requests, MPI_STATUSES_IGNORE);

	// Set exchanged values in P
	if (my_rank > 0) {
		P_halo[0] = from_left;
	}

	if (my_rank < n_ranks-1) {
		P_halo[local_start + n] = from_right;
	}
}












