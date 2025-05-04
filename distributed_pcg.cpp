#include "common.h"

int exchange_P_halo(int my_rank, int n_ranks, int n, Vec& P_halo, MPI_Request reqs[4]) {
    // P_halo layout: [ 0 | 1..n | n+1 ]
    //                ^ghost   real   ^ghost

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

    return cnt;
}

///////////////////////////////////////////////////////////////////


void CG_Solver::init_preconditioner() {

// try the thomas  preconditioner which is supposed
// to be good for our matrix

	//prec.compute(A_block);
	//if (prec.info() != Eigen::Success) {
        //    throw std::runtime_error("CHOL INIT FAILED!");
	//}

   p_a = std::vector<double>(n-1, -1.0);
   p_b = std::vector<double>(n, 2.0);
   p_c = std::vector<double>(n-1, -1.0);
   c_prime = std::vector<double>(n-1);
   d_prime = std::vector<double>(n);
   ms = std::vector<double>(n);


    c_prime[0] = p_c[0] / p_b[0];
    ms[0] = p_b[0];


    for (int i = 1; i < n - 1; ++i) {
        ms[i] = p_b[i] - p_a[i - 1] * c_prime[i - 1];
        c_prime[i] = p_c[i] / ms[i];
    }

    ms[n-1] = p_b[n-1] - p_a[n-2] * c_prime[n-2];
}


void CG_Solver::apply_preconditioner() {

   // r_cond = prec.solve(r);

   // if (prec.info() != Eigen::Success) {
//	throw std::runtime_error("PRECONDITIONER SOLVE FIAILED");
    //}

  // Forward sweep
    d_prime[0] = r[0] / ms[0];

    for (int i=1; i < n-1; ++i) {
        d_prime[i] = (r[i] - p_a[i - 1] * d_prime[i - 1]) / ms[i];
    }

    d_prime[n - 1] = (r[n - 1] - p_a[n - 2] * d_prime[n - 2]) / ms[n-1];

    // Back substitution
    r_cond[n - 1] = d_prime[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        r_cond[i] = d_prime[i] - c_prime[i] * r_cond[i + 1];
    }
    
}

/*
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
*/

// SPLIT INTO LOCAL AND NON LOCAL TO TRY TO OVERLAP
// COMMUNICATION AND COMPUTATION

void CG_Solver::SpMV_local() {
    for (int i=1; i < n-1; ++i) {
        double sum = 0.0;
        for (CSR::InnerIterator it(A, i); it; ++it) {
            int j = it.col() - row_start + 1;
            sum += it.value() * P_halo[j];
        }
        AP[i] = sum;
    }
}

void CG_Solver::SpMV_halo() {
    for (int i : {0, n-1}) {
        double sum = 0.0;
        for (CSR::InnerIterator it(A, i); it; ++it) {
            int j = it.col() - row_start + 1;
            sum += it.value() * P_halo[j];
        }
    AP[i] = sum;
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

                // Initiate exchange with neighbors
                MPI_Request reqs[4];
                int nreqs = exchange_P_halo(my_rank, n_ranks, n, P_halo, reqs);
		//////////////////std::cout << "rank " << my_rank << " exchange" << std::endl;

		// SpMV for local
		SpMV_local();
		//std::cout << "rank " << my_rank << " spvm_local" << std::endl;

                // Now wait for exchanged
                MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);
                //std::cout << "rank " << my_rank << " waiting for exchange" << std::endl;

                // finish SpMV with exchanged
                SpMV_halo();
                //std::cout << "rank " << my_rank << " halo spmv" << std::endl;

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

