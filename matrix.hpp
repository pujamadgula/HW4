#pragma once

#include <mpi.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>


typedef Eigen::SparseMatrix<double, Eigen::RowMajor>     CSR;
typedef Eigen::Triplet<double>                           Triplet;
typedef Eigen::VectorXd                                  Vec;


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
void apply_preconditioner(const CSR& A_block, const Vec& r_local, Vec& z_local) {

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

	z_local = ichol.solve(r_local);

	if (ichol.info() != Eigen::Success) {
		throw std::runtime_error("PRECONDITIONER SOLVE FIAILED");
	}
}



