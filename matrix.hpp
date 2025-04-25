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






