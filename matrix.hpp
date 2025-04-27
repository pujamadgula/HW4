#pragma once

#include <mpi.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>


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
void apply_preconditioner(const CSR& A_block, const Vec& r_local, Vec& r_local_cond) {

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

	r_local_cond = ichol.solve(r_local);

	if (ichol.info() != Eigen::Success) {
		throw std::runtime_error("PRECONDITIONER SOLVE FIAILED");
	}
}



void compute_halo(const CSR A_local,
		  int my_rank, int n_ranks, int N,
		  std::vector<int>& ghost_cols,
		  std::vector<std::vector<int>>& send_ranks,
		  std::vector<std::vector<int>>& recv_ranks,
		  std::unordered_map<int, int>& g2l) {

	int n = A_local.rows();
	int row_start = my_rank * n;
	int row_end = (my_rank + 1) * n;

	// Determine which rank owns the rows of the
	// non-zero columns of this rank's local chunk
	for (int i=0; i < n; ++i) {
		for (CSR::InnerIterator it(A_local, i); it; ++it) {
			int col = it.col();

			if (col < row_start || col >= row_end) {
				if (!g2l.count(col)) {
					int local_index = ghost_cols.size();
					g2l[col] = local_index;
					ghost_cols.push_back(col);
				}
			}
		}
	}


	// Figure out the send and recieves
	recv_ranks.resize(n_ranks);
	send_ranks.resize(n_ranks);

        std::vector<int> send_counts(n_ranks, 0);
	std::vector<int> recv_counts(n_ranks, 0);

	for (int idx=0; idx < (int)ghost_cols.size(); ++idx) {
		int col = ghost_cols[idx];
		int owner = (col * n_ranks) / N;
		recv_ranks[owner].push_back(idx);
	}

	for (int i=0; i < n_ranks; ++i) {
		recv_counts[i] = recv_ranks[i].size();
	}

	MPI_Alltoall(recv_counts.data(), 1, MPI_INT, send_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

	for (int i=0; i < n_ranks; ++i) {
		send_ranks[i].resize(send_counts[i]);
	}
}



void exchange(int my_rank, int n_ranks,
	      const Vec& search_path,
	      const std::vector<int>& ghost_cols,
	      const std::vector<std::vector<int>>& send_ranks,
	      const std::vector<std::vector<int>>& recv_ranks,
	      Vec& p_ghost) {

	int n = search_path.size();
	int ng = ghost_cols.size();

	p_ghost.resize(ng);
	std::vector<MPI_Request> reqs;

	// Queue up a list of recieves we expect
	// do so asynchronously with Irecv
	// will send next
	for (int src=0; src < send_ranks.size(); ++src) {
		if (recv_ranks[src].empty()) continue;

		MPI_Request r;
		MPI_Irecv(p_ghost.data() + recv_ranks[src][0], recv_ranks[src].size(), MPI_DOUBLE, src, 0, MPI_COMM_WORLD, &r);

		reqs.push_back(r);
	}


	// prepare send buffers
	std::vector<std::vector<double>> sendbufs(n_ranks);

	for (int dst=0; dst < n_ranks; ++dst) {
		if (send_ranks[dst].empty()) continue;

		for (int local_idx : send_ranks[dst]) {
			sendbufs[dst].push_back(search_path[local_idx]);
		}
	}


	// recvs
	for (int src=0; src < n_ranks; ++src) {
		if (recv_ranks[src].empty()) continue;

		MPI_Request r;


		std::cout << "Rank " << my_rank << " Irecv " << recv_ranks[src].size() << " from rank " << src << std::endl;


		MPI_Irecv(p_ghost.data() + recv_ranks[src][0], recv_ranks[src].size(), MPI_DOUBLE, src, 0, MPI_COMM_WORLD, &r);

		reqs.push_back(r);
	}

	// sends
	for (int dst=0; dst < n_ranks; ++dst) {
		if (send_ranks[dst].empty()) continue;

		MPI_Request s;

		std::cout << "Rank " << my_rank << " Isend " << sendbufs[dst].size() << " to rank " << dst << std::endl;

		MPI_Isend(sendbufs[dst].data(), sendbufs[dst].size(), MPI_DOUBLE, dst, 0, MPI_COMM_WORLD, &s);

		reqs.push_back(s);
	}


	// Wait on all send and recieves
	MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
}



// Matrix - Vector Multiply on local chunk
// doesnt do any optimization rn
// just the naive way
void local_m_v(const CSR& A_local,
	       const Vec& search_path,
	       const Vec& p_ghost,
	       const std::unordered_map<int, int>& g2l,
	       Vec& result) {

	int n = A_local.rows();
	result.setZero(n);

	for (int i=0; i < n; ++i) {
		double sum = 0;
		for (CSR::InnerIterator it(A_local, i); it; ++it) {
			int col = it.col();
			double v = it.value();

			if (g2l.count(col)) {
				sum += v * p_ghost[g2l.at(col)];
			} else {
				sum += v * search_path[col % n];
			}
		}
		result[i] = sum;
	}
}


	       
	


