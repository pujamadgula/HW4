#pragma once

#include <mpi.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <iostream>

typedef Eigen::SparseMatrix<double, Eigen::RowMajor>     CSR;
typedef Eigen::Triplet<double>                           Triplet;
typedef Eigen::VectorXd                                  Vec;
typedef Eigen::VectorBlock<Eigen::VectorXd>              VecView;


class CG_Solver {
    public:

	int my_rank, n_ranks, n,N, row_start, row_end;
	CSR A, A_block;
	Vec x, b, r, r_cond, P_halo, AP;
	//Eigen::IncompleteCholesky<double, Eigen::Lower, Eigen::NaturalOrdering<int>> prec;
	double prev_rr_local, prev_rr;


        std::vector<double> p_a, p_b, p_c, c_prime, d_prime, ms;

	//timings
	double exchange_func_time, spmv_local_time, wait_time, spmv_halo_time, alpha_time, update_time, preconditioner_time, residual_time, beta_time, P_update_time, copy_solution_time;

	CG_Solver(int _n, int _N);
	bool solve(std::vector<double>& solution, int max_iters, double tol);

        void init_preconditioner();
        void apply_preconditioner();
        void SpMV_local();
        void SpMV_halo();

};




