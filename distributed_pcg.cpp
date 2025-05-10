#include "common.h"
    
#include <mpi.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <iostream>

typedef Eigen::SparseMatrix<double, Eigen::RowMajor>     CSR;
typedef Eigen::Triplet<double>                           Triplet;
typedef Eigen::VectorXd                                  Vec;
typedef Eigen::VectorBlock<Eigen::VectorXd>              VecView;


class i_CG_Solver {
    public:

	int my_rank, n_ranks, n,N, row_start, row_end;
	CSR A, A_block;
	Vec x, b, r, r_cond, P_halo, AP;
	//Eigen::IncompleteCholesky<double, Eigen::Lower, Eigen::NaturalOrdering<int>> prec;
	double prev_rr_local, prev_rr;


        // std::vector<double> p_a, p_b, p_c, c_prime, d_prime, ms, inv_ms;
        double *p_a, *p_b, *p_c, *c_prime, *d_prime, *ms, *inv_ms;

	//timings
	//double exchange_func_time, spmv_local_time, wait_time, spmv_halo_time, alpha_time, update_time, preconditioner_time, residual_time, beta_time, P_update_time, copy_solution_time;

	i_CG_Solver();
	void init(int _n, int _N);
	~i_CG_Solver();
	void solve(std::vector<double>& solution, int max_iters, double tol);

        void init_preconditioner();
        void apply_preconditioner();
        void SpMV_local();
        void SpMV_halo();

};




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


void i_CG_Solver::init_preconditioner() {

// try the thomas  preconditioner which is supposed
// to be good for our matrix

	//prec.compute(A_block);
	//if (prec.info() != Eigen::Success) {
        //    throw std::runtime_error("CHOL INIT FAILED!");
	//}


    c_prime[0] = p_c[0] / p_b[0];
    ms[0] = p_b[0];


    for (int i = 1; i < n - 1; ++i) {
        ms[i] = p_b[i] - p_a[i - 1] * c_prime[i - 1];
        c_prime[i] = p_c[i] / ms[i];
    }

    ms[n-1] = p_b[n-1] - p_a[n-2] * c_prime[n-2];

    // pre-divide and spare divison in each
    // cycle of the cj steps
    for(int i=0; i<n; ++i)
        inv_ms[i] = 1.0/ms[i];

}


void i_CG_Solver::apply_preconditioner() {

    // — Forward sweep —  
    double prev = r[0] * inv_ms[0];  
    d_prime[0] = prev;  

    for(int i = 1; i < n; ++i) {  
        double t = r[i] - p_a[i-1] * prev;  
        prev    = t * inv_ms[i];  
        d_prime[i]   = prev;  
    }  

    // — Back substitution —  
    double next = d_prime[n-1];  
    r_cond[n-1] = next;  

    for(int i = n-2; i >= 0; --i) {  
        double t = d_prime[i] - c_prime[i] * next;  
        next    = t;  
        r_cond[i]     = t;  
    }  
}



// SPLIT INTO LOCAL AND NON LOCAL TO TRY TO OVERLAP
// COMMUNICATION AND COMPUTATION

void i_CG_Solver::SpMV_local() {
     /* 
      for (int i=1; i < n-1; ++i) {
        double sum = 0.0;
        for (CSR::InnerIterator it(A, i); it; ++it) {
            int j = it.col() - row_start + 1;
            sum += it.value() * P_halo[j];
        }
        AP[i] = sum;
    }
    */
    
    for(int i=0; i < n; ++i){
      AP[i] = 2.0*P_halo[i+1] - P_halo[i] - P_halo[i+2];
    }
    
}

void i_CG_Solver::SpMV_halo() {
    /*
    for (int i : {0, n-1}) {
        double sum = 0.0;
        for (CSR::InnerIterator it(A, i); it; ++it) {
            int j = it.col() - row_start + 1;
            sum += it.value() * P_halo[j];
        }
    AP[i] = sum;
    }
    */

    if(row_start > 0) {
      AP[0] = -P_halo[0] + 2.0*P_halo[1] - P_halo[2];
    } else {
      // no left neighbor
      AP[0] =  2.0*P_halo[1] - P_halo[2];
    }

    // last row (i==n-1)
    if(row_start + (n-1) < N-1) {
      AP[n-1] = -P_halo[n-1] + 2.0*P_halo[n] - P_halo[n+1];
    } else {
      // no right neighbor
      AP[n-1] = -P_halo[n-1] + 2.0*P_halo[n];
    }

}


////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
i_CG_Solver::i_CG_Solver() {};

void i_CG_Solver::init(int _n, int _N) {

	// exchange_func_time, spmv_local_time, wait_time, spmv_halo_time, alpha_time, update_time, preconditioner_time, residual_time, beta_time, P_update_time, copy_solution_time = 0.0;

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

       // Allocate for the preconditioner
       p_a = new double[n-1];
       std::fill_n(p_a, n-1, -1.0);

       p_b = new double[n];
       std::fill_n(p_b, n, 2.0);

       p_c = new double[n-1];
       std::fill_n(p_c, n-1, -1.0);

       c_prime = new double[n-1];
       d_prime = new double[n];
       ms = new double[n];
       inv_ms = new double[n];



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

i_CG_Solver::~i_CG_Solver() {
       delete[] p_a;
       delete[] p_b;

       delete[] p_c;

       delete[] c_prime;
       delete[] d_prime;
       delete[] ms;
       delete[] inv_ms;
}


void i_CG_Solver::solve(std::vector<double>& solution, int max_iters, double tol) {

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
		
		// double exchange_func_start = MPI_Wtime();
                MPI_Request reqs[4];
                int nreqs = exchange_P_halo(my_rank, n_ranks, n, P_halo, reqs);
		//exchange_func_time += MPI_Wtime() - exchange_func_start;
		//////////////////std::cout << "rank " << my_rank << " exchange" << std::endl;

		// SpMV for local
		//double spmv_local_start = MPI_Wtime();
		SpMV_local();
		//spmv_local_time += MPI_Wtime() - spmv_local_start;
		//std::cout << "rank " << my_rank << " spvm_local" << std::endl;

                // Now wait for exchanged
		//double wait_start = MPI_Wtime();
                MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);
		//wait_time += MPI_Wtime() - wait_start;
                //std::cout << "rank " << my_rank << " waiting for exchange" << std::endl;

                // finish SpMV with exchanged
		//double spmv_halo_start = MPI_Wtime();
                SpMV_halo();
		//spmv_halo_time += MPI_Wtime() - spmv_halo_start;
                //std::cout << "rank " << my_rank << " halo spmv" << std::endl;

		// alpha
		//double alpha_start = MPI_Wtime();
		local_pap = P.dot(AP);
		MPI_Allreduce(&local_pap, &global_pap, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		alpha = prev_rr / global_pap;
		//alpha_time += MPI_Wtime() - alpha_start;
		//std::cout << "rank " << my_rank << " alpha" << std::endl;

		// Update solution and residual
		//double update_start = MPI_Wtime();
		x += alpha * P;
		r -= alpha * AP;
		//update_time += MPI_Wtime() - update_start;
		//std::cout << "rank " << my_rank << " update x/r" << std::endl;

		// Precondition new residual
		//double preconditioner_start = MPI_Wtime();
		apply_preconditioner();
		//preconditioner_time += MPI_Wtime() - preconditioner_start;

		//double residual_start  = MPI_Wtime();
		new_rr_local = r.dot(r_cond);
		MPI_Allreduce(&new_rr_local, &new_rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		//std::cout << "rank " << my_rank << " precondition new r" << std::endl;

		// check residual norm
		res_norm = std::sqrt(new_rr);
		//residual_time += MPI_Wtime() - residual_start;


		if (res_norm < epsilon) {
		        //std::cout << "rank " << my_rank << " breaking" << std::endl;
			converged = true;
			break;
		}

		// beta
		//double beta_start = MPI_Wtime();
		beta = new_rr / prev_rr;
		prev_rr = new_rr;
		//beta_time += MPI_Wtime() - beta_start;
		//std::cout << "rank " << my_rank << " beta" << std::endl;

		// update search path
		//double P_update_start = MPI_Wtime();
		P = r_cond + beta * P;
		//P_update_time += MPI_Wtime() - P_update_start;
		//std::cout << "rank " << my_rank << " update p" << std::endl;
	}

	// copy result to solution vector
	//double copy_solution_start = MPI_Wtime();
	Vec::Map(solution.data(), x.size()) = x;
	//copy_solution_time += MPI_Wtime() - copy_solution_start;
}



////////////////////////////////////////////////////////////////////
i_CG_Solver icg;
///////////////////////////////////////////////////////////////////

CG_Solver::CG_Solver(const int& n, const int& N) {
	int _n = n;
	int _N = N;
	icg.init(_n, _N);
}

void CG_Solver::solve(const std::vector<double>& b, std::vector<double>& x, double tol) {
	icg.solve(x, 10000, tol);
}

















