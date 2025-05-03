#include "common.h"

void exchange_P_halo(int my_rank, int n_ranks, int n, Vec& P_halo) {
    // P_halo layout: [ 0 | 1..n | n+1 ]
    //                ^ghost   real   ^ghost

    MPI_Request reqs[4];
    int cnt = 0;

    // send of first real to left neighbour’s right ghost
    if (my_rank > 0) {
        MPI_Isend(&P_halo[1],      1, MPI_DOUBLE, my_rank-1, 0, MPI_COMM_WORLD, &reqs[cnt++]);
        MPI_Irecv(&P_halo[0],      1, MPI_DOUBLE, my_rank-1, 0, MPI_COMM_WORLD, &reqs[cnt++]);
    }

    // send of last real to right neighbour’s left ghost
    if (my_rank < n_ranks-1) {
        MPI_Isend(&P_halo[n],      1, MPI_DOUBLE, my_rank+1, 0, MPI_COMM_WORLD, &reqs[cnt++]);
        MPI_Irecv(&P_halo[n+1],    1, MPI_DOUBLE, my_rank+1, 0, MPI_COMM_WORLD, &reqs[cnt++]);
    }

    MPI_Waitall(cnt, reqs, MPI_STATUSES_IGNORE);

}

///////////////////////////////////////////////////////////////////


void CG_Solver::init_preconditioner() {

	ichol.compute(A_block);
	if (ichol.info() != Eigen::Success) {
            throw std::runtime_error("CHOL INIT FAILED!");
	}
}


void CG_Solver::apply_preconditioner() {

    r_cond = ichol.solve(r);

    if (ichol.info() != Eigen::Success) {
	throw std::runtime_error("PRECONDITIONER SOLVE FIAILED");
    }
}

void CG_Solver::SpMV() {

    for (int local_row = 0; local_row < n; ++local_row) {
        AP[local_row] = 0;

        for (CSR::InnerIterator it(A, local_row); it; ++it) {
            int global_col = it.col();
	    int P_idx = global_col - row_start + 1;

            AP[local_row] += it.value() * P_halo[P_idx];
        }
    }
}


////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

CG_Solver::CG_Solver(int _n, int _N) {
	n = _n;
	N = _N;

	MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	// Initialize our chunk of A
	A = CSR(n, N);
	std::vector<Triplet> triplets;

        row_start = my_rank * n;
        row_end = (my_rank+1) * n;

	for (int i=0; i < n; ++i) {
		int gi = row_start + i;

		if (gi > 0) triplets.emplace_back(i, gi-1, -1.0);

		triplets.emplace_back(i, gi, 2.0);

		if (gi < N-1) triplets.emplace_back(i, gi+1, -1.0);

	}

	A.setFromTriplets(triplets.begin(), triplets.end());
	A.makeCompressed();

	// Get our diagonal block of A
	A_block = CSR(n,n);
	triplets.clear();

	for (int i=0; i < n; ++i) {
		for (CSR::InnerIterator it(A, i); it; ++it) {
			int col = it.col();

			if (col >= row_start && col < row_start + n) {
				int local_col = col - row_start;
				triplets.emplace_back(i, local_col, it.value());
			}
		}
	}

	A_block.setFromTriplets(triplets.begin(), triplets.end());
	A_block.makeCompressed();

	// Initialize the preconditioner
	// on our A_block
	init_preconditioner();

	// Initialize our vectors
	x = Vec::Zero(n);
	b = Vec::Ones(n);
	r = Vec::Ones(n);
	r_cond = Vec(n);
	AP = Vec(n);

	// Apply initial preconditioning
	apply_preconditioner();
	prev_rr_local = r.dot(r_cond);
	MPI_Allreduce(&prev_rr_local, &prev_rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	// Set initial search direction
        P_halo = Vec::Zero(n+2);
	P_halo.segment(1, n) = r_cond;

	//std::cout << "rank " << my_rank << " finished init" << std::endl;
}


bool CG_Solver::solve(std::vector<double>& solution, int max_iters, double tol) {

	// Stopping criteria
	double epsilon = tol * std::sqrt(b.dot(b));

	// return a bool for whether it converged
	bool converged = false;

        // View of P_halo that is actually
	// owned by this rank
	VecView P = P_halo.segment(1, n);

	double local_pap, global_pap;
	double alpha;

	double new_rr_local, new_rr;
	double res_norm, beta;

	for (int iter=0; iter < max_iters; iter++) {

		// Exchange with neighbors
		exchange_P_halo(my_rank, n_ranks, n, P_halo);
		//////////////////std::cout << "rank " << my_rank << " exchange" << std::endl;

		// SpMV
		SpMV();
		//std::cout << "rank " << my_rank << " spvm" << std::endl;

		// alpha
		local_pap = P.dot(AP);
		MPI_Allreduce(&local_pap, &global_pap, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		alpha = prev_rr / global_pap;
		//std::cout << "rank " << my_rank << " alpha" << std::endl;

		// Update solution and residual
		x += alpha * P;
		r -= alpha * AP;
		//std::cout << "rank " << my_rank << " update x/r" << std::endl;

		// Precondition new residual
		apply_preconditioner();
		new_rr_local = r.dot(r_cond);
		MPI_Allreduce(&new_rr_local, &new_rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		//std::cout << "rank " << my_rank << " precondition new r" << std::endl;

		// check residual norm
		res_norm = std::sqrt(new_rr);
		if (res_norm < epsilon) {
		        //std::cout << "rank " << my_rank << " breaking" << std::endl;
			converged = true;
			break;
		}

		// beta
		beta = new_rr / prev_rr;
		prev_rr = new_rr;
		//std::cout << "rank " << my_rank << " beta" << std::endl;

		// update search path
		P = r_cond + beta * P;
		//std::cout << "rank " << my_rank << " update p" << std::endl;
	}

	// copy result to solution vector
	Vec::Map(solution.data(), x.size()) = x;
	return converged;
}

