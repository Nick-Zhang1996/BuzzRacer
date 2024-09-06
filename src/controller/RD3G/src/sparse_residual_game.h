#pragma once
//#define EIGEN_RUNTIME_NO_MALLOC
//Eigen::internal::set_is_malloc_allowed(false);

#include <iostream>
#include <fstream>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SparseCore>
#include <stdexcept>
#include <string>

// get virtual memory currently used by this process
//#include "stdlib.h"
//#include "stdio.h"
//#include "string.h"

// sparse solvers
#include <Eigen/OrderingMethods>
#include <Eigen/SparseQR>
// for LeastSquaresConjugateGradient
#include<Eigen/IterativeLinearSolvers>

#include "profiler.h"


// TODO fix h_plus_sum re-calculation
// TODO add fill-in style api
// TODO add solve
// TODO make program self-independent
// TODO use template format for block
// TODO remove temporary variables?

using Scalar = double;
using std::endl;
using std::cout;
using std::min;
using Eigen::MatrixBase;
// NOTE has to be RowMajor
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
// SpMatrix was taken
using SpMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor>;

template <typename Derived>
void checksum(const MatrixBase<Derived>& mtx){
  Eigen::Index  maxIndex;
  float maxNorm = mtx.rowwise().sum().maxCoeff(&maxIndex);
  std::cout << "Maximum sum at position " << maxIndex << std::endl;
  std::cout << "its sum is is: " << maxNorm << std::endl;

  Eigen::Index  minIndex;
  float minNorm = mtx.rowwise().sum().minCoeff(&minIndex);
  std::cout << "Minimum sum at position " << minIndex << std::endl;
  std::cout << "its sum is is: " << minNorm << std::endl;

}

inline double sqr(const double a){
    return a*a;
}

int getCurrentMemoryUsageInKB(){ //Note: this value is in KB!
    std::ifstream file("/proc/self/status");
    std::string line;
    int memory_usage = 0;

    while (std::getline(file,line)){
        if (line.rfind("VmSize",0) == 0){
            std::size_t startPos = line.find_first_of("0123456789");
            std::size_t endPos = line.find(" kB");
            memory_usage = std::stoi(line.substr(startPos, endPos - startPos));
            break;
        }
    }
    return memory_usage;
}

// assign a dense matrix to a sub-block of a sparse matrix
// this function assumes there's no existing entries in the sparse matrix, it uses SpMatrix.insert()
// for addition, use sp_add()
template <typename Derived>
void sp_assign(const MatrixBase<Derived>& in_mtx, SpMatrix& out_mtx, const int row_offset, const int col_offset, const int row_size, const int col_size){
    // maybe we can avoid creating this variable?
    const SpMatrix sp_in_mtx = in_mtx.sparseView();
    for (int k=0; k<sp_in_mtx.outerSize(); ++k){
        for (SpMatrix::InnerIterator it(sp_in_mtx,k); it; ++it){
            out_mtx.insert(row_offset + it.row(), col_offset + it.col()) = it.value();
            // we should check that we are not inserting outside the target block, creating a memory leak, but only for debug build
        }
    }
}

template <typename Derived>
void sp_add(const MatrixBase<Derived>& in_mtx, SpMatrix& out_mtx, const int row_offset, const int col_offset, const int row_size, const int col_size){
    // maybe we can avoid creating this variable?
    const SpMatrix sp_in_mtx = in_mtx.sparseView();
    for (int k=0; k<sp_in_mtx.outerSize(); ++k){
        for (SpMatrix::InnerIterator it(sp_in_mtx,k); it; ++it){
            out_mtx.coeffRef(row_offset + it.row(), col_offset + it.col()) += it.value();
            // we should check that we are not inserting outside the target block, creating a memory leak, but only for debug build
        }
    }
}

std::tuple<SpMatrix,std::vector<int>> remove_empty_cols(SpMatrix& matrix, const int reserve_size) {
    //  Identify non-empty columns
    std::vector<int> nonEmptyCols;
    nonEmptyCols.reserve(reserve_size);
    for (int j = 0; j < matrix.cols(); ++j) {
        if (matrix.col(j).nonZeros() > 0) {
            nonEmptyCols.push_back(j);
        }
    }

    //  Create a new temporary matrix with non-empty columns
    SpMatrix tempMatrix(matrix.rows(), nonEmptyCols.size());
    for (int newColIdx = 0; newColIdx < nonEmptyCols.size(); ++newColIdx) {
        int oldColIdx = nonEmptyCols[newColIdx];
        tempMatrix.col(newColIdx) = matrix.col(oldColIdx);
    }

    return {tempMatrix,nonEmptyCols};
}


template <int n, int m>
class ResidualGame {

    protected:
        int N,T;
        Scalar dt,rho,rho_b,bc_a,bc_b;
        Scalar tolerance;
        int backtracking_max_iter;
        Matrix x0;
        Profiler<true> profiler;
        int current_memory_usage_kb;

    public:
        ResidualGame(const int _N, const int _T,
                const Scalar _dt, const Scalar _rho, const Scalar _rho_b, const Scalar _bc_a, const Scalar _bc_b, const Scalar _tolerance, const int _backtracking_max_iter):
            N(_N), T(_T),
            dt(_dt), rho(_rho), rho_b(_rho_b),bc_a(_bc_a), bc_b(_bc_b),
            tolerance(_tolerance), backtracking_max_iter(_backtracking_max_iter),
            x0(),profiler(),current_memory_usage_kb(0) {
                //current_memory_usage_kb = getCurrentMemoryUsageInKB();
                //std::cout << "existing memory usage " << current_memory_usage_kb << "KB" << std::endl;
        }

        void set_x0(const Matrix &val){
            x0 = Matrix(val);
        }
        void post_step_update(){
            rho *= rho_b;
        }

        // x_k: dim: N*n, u_k_i: dim:m*1, lambda_k:N*n, h_k_plus_mask: N*N, mu_k dim:N*N
        Matrix dL_dx_ik(const Matrix x_k,const Matrix  u_k_i,const Matrix  x_k1_i,const Matrix h_k_plus_mask,const Matrix lamda_k,const Matrix mu_k,const int i){
            Matrix val =  dJi_dxi(x_k,u_k_i,i) + lamda_k.row(i) * df_dx(x_k.row(i).transpose(),u_k_i,i);
            for (int j=0; j<N; j++){
                if (i==j){continue;}
                if (h_k_plus_mask(i,j)){
                    val +=  mu_k(i,j) * ( dh_dxi(x_k.row(i).transpose(), x_k.row(j).transpose()) );
                } else {
                    val += -1.0/rho*min(1.0/h(x_k.row(i).transpose(), x_k.row(j).transpose()),1e10) * dh_dxi(x_k.row(i).transpose(), x_k.row(j).transpose());
                }
            }
            return val;
        }

        Matrix dL_du(const Matrix x_k, const Matrix u_k_i, const Matrix x_k1_i, const Matrix h_k_plus_mask,const Matrix lamda_k, const Matrix mu_k,const int i){
            return dJi_du(x_k,u_k_i,i) + lamda_k.row(i) * df_du(x_k.row(i).transpose(), u_k_i,i);
        }

        // for 3d array, first dimension is list() -> std::vector
        //x_i_k: 1..T, T*N*n  NOTE starts from 1
        //u_i_k: 0..T-1, T*N*m
        //lamda_i_k: 0..T-1 T*N*n
        //mu_k_i_j: 1..T T*N*N NOTE starts from 1
        Matrix dLLi_dxi(const std::vector<Matrix>& x,const std::vector<Matrix>& u,const std::vector<Matrix>& h_plus_mask,const std::vector<Matrix>& lamda,const std::vector<Matrix>& mu,const int i){
            // TODO is this the best approach?
            Matrix der(1,T*n);
            der.setZero();
            // dLLi_dxi
            for (int k=1; k<T; k++){
                der.template block<1,n>(0,(k-1)*n) = dL_dx_ik(x[k-1],u[k,i],x[k].row(i).transpose(),h_plus_mask[k-1],lamda[k],mu[k-1],i) -lamda[k-1].row(i);
            }
            // dLLi_dxi_T
            der.template block<1,n>(0,(T-1)*n) = -lamda[T-1].row(i) + dJi_dxi(x[T-1],Matrix::Zero(m,1),i);
            for (int j=0; j<N; j++){
                if (i==j){continue;}
                if (h_plus_mask[T-1](i,j)){
                der.template block<1,n>(0,(T-1)*n) +=  mu[T-1](i,j) *  dh_dxi(x[T-1].row(i).transpose(), x[T-1].row(j).transpose());
                } else {
                der.template block<1,n>(0,(T-1)*n) += -1/rho*min(1.0/h(x[T-1].row(i).transpose(), x[T-1].row(j).transpose()),1e10)*dh_dxi(x[T-1].row(i).transpose(),x[T-1].row(j).transpose());
                }
            }
            return der;
        }

        // for 3d array, first dimension is list() -> std::vector
        //x_i_k: 1..T, T*N*n  NOTE starts from 1
        //u_i_k: 0..T-1, T*N*m
        //lamda_i_k: 0..T-1 T*N*n
        //mu_k_i_j: 1..T T*N*N NOTE starts from 1
        // TODO obsolete
        Matrix dLLi_dx(const std::vector<Matrix>& x,const std::vector<Matrix>& u,const std::vector<Matrix>& h_plus_mask,const std::vector<Matrix>& lamda,const std::vector<Matrix>& mu,const int i){
            throw std::runtime_error("obsolete function called");

            Matrix der(1,T*N*n);
            der.setZero();
            // dLLi_dxi
            for (int k=1; k<T; k++){
                der.template block<1,n>(0,(k-1)*N*n+i*n) = dL_dx_ik(x[k-1],u[k,i],x[k].row(i).transpose(),h_plus_mask[k-1],lamda[k],mu[k-1],i) -lamda[k-1].row(i);
            }
            // dLLi_dxi_T
            der.template block<1,n>(0,(T-1)*N*n+i*n) = -lamda[T-1].row(i) + dJi_dxi(x[T-1],Matrix::Zero(m,1),i);
            for (int j=0; j<N; j++){
                if (i==j){continue;}
                if (h_plus_mask[T-1](i,j)){
                der.template block<1,n>(0,(T-1)*N*n+i*n) +=  mu[T-1](i,j) *  dh_dxi(x[T-1].row(i).transpose(), x[T-1].row(j).transpose());
                } else {
                der.template block<1,n>(0,(T-1)*N*n+i*n) += -1/rho*min(1.0/h(x[T-1].row(i).transpose(), x[T-1].row(j).transpose()),1e10)*dh_dxi(x[T-1].row(i).transpose(),x[T-1].row(j).transpose());
                }
            }
            // dLLi_dxj
            for (int j=0; j<N; j++){
                if (i==j){continue;}
                for (int k=1; k<T; k++){
                    if(h_plus_mask[k-1](i,j)){
                        der.template block<1,n>(0,(k-1)*N*n+j*n) = mu[k-1](i,j) * dh_dxj(x[k-1].row(i).transpose(), x[k-1].row(j).transpose());
                    } else {
                        der.template block<1,n>(0,(k-1)*N*n+j*n) = -1.0/rho*min(1.0/h(x[k-1].row(i).transpose(), x[k-1].row(j).transpose()),1e10)*dh_dxj(x[k-1].row(i).transpose(), x[k-1].row(j).transpose());
                    }
                    der.template block<1,n>(0,(k-1)*N*n+j*n) += dJi_dxj(x[k-1],u[k].row(i).transpose(), i, j);
                }
                const int k = T;
                if(h_plus_mask[k-1](i,j)){
                    der.template block<1,n>(0,(k-1)*N*n+j*n) = mu[k-1](i,j) * dh_dxj(x[k-1].row(i).transpose(), x[k-1].row(j).transpose());
                } else {
                    der.template block<1,n>(0,(k-1)*N*n+j*n) = -1.0/rho*min(1.0/h(x[k-1].row(i).transpose(), x[k-1].row(j).transpose()),1e10)*dh_dxj(x[k-1].row(i).transpose(), x[k-1].row(j).transpose());
                }
                der.template block<1,n>(0,(k-1)*N*n+j*n) += dJi_dxj(x[k-1],Matrix::Zero(m,1), i, j);

            }
            return der;
        }

        Matrix dLLi_dui(const std::vector<Matrix>& x, const std::vector<Matrix>& u, const std::vector<Matrix>& h_plus_mask, const std::vector<Matrix>& lamda, const std::vector<Matrix>& mu, const int i) {
            Matrix der(1, T * m); // Initialize derivative vector as row vector
            der.setZero(); // Ensure all items are properly zero-initialized

            // dLLi_dui_0
            der.template block<1,m>(0,  0) = dJi_du(x0, u[0].row(i).transpose(), i) + lamda[0].row(i) * df_du(x0.row(i).transpose(), u[0].row(i).transpose(), i);

            // dLLi_dui_k
            for (int k = 1; k < T; ++k) {
                der.template block<1,m>(0, k * m) = dJi_du(x[k - 1], u[k].row(i).transpose(), i) + lamda[k].row(i) * df_du(x[k - 1].row(i).transpose(), u[k].row(i).transpose(),i);
            }

            return der;
        }

        // TODO obsolete
        Matrix dLLi_du(const std::vector<Matrix>& x, const std::vector<Matrix>& u, const std::vector<Matrix>& h_plus_mask, const std::vector<Matrix>& lamda, const std::vector<Matrix>& mu, const int i) {
            throw std::runtime_error("obsolete function called");
            Matrix der(1, T * N* m); // Initialize derivative vector as row vector
            der.setZero(); // Ensure all items are properly zero-initialized

            // dLLi_dui_0
            der.template block<1,m>(0, i * m) = dJi_du(x0, u[0].row(i).transpose(), i) + lamda[0].row(i) * df_du(x0.row(i).transpose(), u[0].row(i).transpose(), i);

            // dLLi_dui_k
            for (int k = 1; k < T; ++k) {
                der.template block<1,m>(0, i * m + k * N * m) = dJi_du(x[k - 1], u[k].row(i).transpose(), i) + lamda[k].row(i) * df_du(x[k - 1].row(i).transpose(), u[k].row(i).transpose(),i);
            }

            return der;
        }

        Matrix dBh_dxi(const Matrix& x_i, const Matrix& x_j) {
            // Calculate dBh/dxi
            return -1.0 / (rho * h(x_i, x_j)) * dh_dxi(x_i, x_j);
        }

        Matrix dBh_dxj(const Matrix& x_i, const Matrix& x_j) {
            // Calculate dBh/dxj
            return -1.0 / (rho * h(x_i, x_j)) * dh_dxj(x_i, x_j);
        }
        Matrix dBh_dxi_dxi(const Matrix& x_i, const Matrix& x_j) {
            // Calculate h, dh/dxi, and dhdxi_dxi
            Scalar h_val = h(x_i, x_j);
            Matrix dhdxi = dh_dxi(x_i, x_j);
            Matrix dhdxi_dxi = dh_dxi_dxi(x_i, x_j);

            // Calculate dBh/dxi_dxi
            Matrix val = 1.0 / (rho * h_val) * (-dhdxi_dxi + 1.0 / h_val * dhdxi.transpose() * dhdxi);
            return val;
        }
        Matrix dBh_dxi_dxj(const Matrix& x_i, const Matrix& x_j) {
            // Calculate h, dh/dxi, and dhdxi_dxi
            Scalar h_val = h(x_i, x_j);
            Matrix dhdxi = dh_dxi(x_i, x_j);
            Matrix dhdxj = dh_dxj(x_i, x_j);
            Matrix dhdxi_dxj = dh_dxi_dxj(x_i, x_j);

            // Calculate dBh/dxi_dxi
            Matrix val = 1.0 / (rho * h_val) * (-dhdxi_dxj + 1.0 / h_val * dhdxi.transpose() * dhdxj);
            return val;
        }
        Matrix dBh_dxj_dxj(const Matrix& x_i, const Matrix& x_j) {
            // Calculate h, dh/dxi, and dhdxi_dxi
            Scalar h_val = h(x_i, x_j);
            Matrix dhdxj = dh_dxj(x_i, x_j);
            Matrix dhdxj_dxj = dh_dxj_dxj(x_i, x_j);

            // Calculate dBh/dxi_dxi
            Matrix val = 1.0 / (rho * h_val) * (-dhdxj_dxj + 1.0 / h_val * dhdxj.transpose() * dhdxj);
            return val;
        }
        Matrix dF_dx(const std::vector<Matrix>& x, const std::vector<Matrix>& u, const int i, const int k) {
            // Initialize dF_dx matrix
            Matrix dFdx = Matrix::Zero(n, T * N * n);

            // Calculate indices for insertion
            int start_idx_1 = (k - 1) * N * n + i * n;
            int start_idx_2 = k * N * n + i * n;

            // Assign values to dF_dx
            dFdx.template block<n,n>(0, start_idx_1) = df_dx(x[k - 1].row(i).transpose(), u[k].row(i).transpose(),i);
            dFdx.template block<n,n>(0, start_idx_2) = -Matrix::Identity(n, n);

            return dFdx;
        }

        Matrix dF0_dx(const std::vector<Matrix>& x, const std::vector<Matrix>& u, const int i) {
            // Initialize dF_dx matrix
            Matrix dFdx = Matrix::Zero(n, T * N * n);
            dFdx.template block<n,n>(0, i*n) = -Matrix::Identity(n, n);

            return dFdx;
        }

        Matrix dh_dx(const std::vector<Matrix>& x, const int k, const int i, const int j) {
            // Initialize dh_dx matrix
            Matrix dhdx = Matrix::Zero(1, T * N * n);

            // Calculate indices for insertion
            int start_idx_1 = (k - 1) * N * n + i * n;
            int start_idx_2 = (k - 1) * N * n + j * n;

            // Assign values to dh_dx
            dhdx.template block<1,n>(0, start_idx_1) = dh_dxi(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());
            dhdx.template block<1,n>(0, start_idx_2) = dh_dxj(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());

            return dhdx;
        }

        Matrix dLLi_dxi_dx(const std::vector<Matrix>& x, const std::vector<Matrix>& u, const std::vector<Matrix>& h_plus_mask, const std::vector<Matrix>& lamda, const std::vector<Matrix>& mu, const int i) {
            //cout << "dLLi_dxdx " << endl;
            Matrix dLL_dxi_dx = Matrix::Zero(T*n, T*N*n);

            auto submtx = [&](int k, int j) {
                return dLL_dxi_dx.template block<n,n>((k - 1) * n, (k - 1) * N * n + j * n);
            };

            // dJi_dxi_dxi
            //cout << "dJi_dxi_dxi " << endl;
            for (int k = 1; k < T; ++k) {
                auto mtx = submtx(k, i);
                mtx = dJi_dxi_dxi(x[k - 1], u[k].row(i).transpose(), i);
                for (int j=0; j<N; ++j){
                    if (i==j){continue;}
                    if (h_plus_mask[k-1](i,j)){
                        mtx += mu[k - 1](i,j) * dh_dxi_dxi(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());
                    } else {
                        mtx += dBh_dxi_dxi(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());
                    }
                }
            }

            // dJi_dxi_dxi, k = T
            //cout << "dJi_dxi_dxi k=T " << endl;
            auto mtx = submtx(T, i);
            mtx = dJi_dxi_dxi(x[T - 1], Matrix::Zero(m,1), i);
            for (int j=0; j<N; ++j){
                if (i==j){continue;}
                if (h_plus_mask[T-1](i,j)){
                    mtx += mu[T - 1](i,j) * dh_dxi_dxi(x[T - 1].row(i).transpose(), x[T - 1].row(j).transpose());
                } else {
                    mtx += dBh_dxi_dxi(x[T - 1].row(i).transpose(), x[T - 1].row(j).transpose());
                }
            }

            // dJi_dxi_dxj
            //cout << "dJi_dxi_dxj " << endl;
            for (int k = 1; k < T; ++k) {
                for (int j = 0; j < N; ++j) {
                    if (i == j) {
                        continue;
                    }
                    Matrix val = Matrix::Zero(n, n);
                    if (h_plus_mask[k-1](i,j)){
                        val = dJi_dxi_dxj(x[k-1], u[k].row(i).transpose(), i, j) + mu[k - 1](i,j) * dh_dxi_dxj(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());
                    } else {
                        val = dJi_dxi_dxj(x[k-1], u[k].row(i).transpose(), i, j) + dBh_dxi_dxj(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());
                    }
                    submtx(k,j) = val;
                }
            }

            //cout << "dJi_dxi_dxj k=T" << endl;
            const int k = T;
            for (int j = 0; j < N; ++j) {
                if (i == j) {
                    continue;
                }
                Matrix val = Matrix::Zero(n, n);
                if (h_plus_mask[k-1](i,j)){
                    val = dJi_dxi_dxj(x[k-1], Matrix::Zero(m,1), i, j) + mu[k - 1](i,j) * dh_dxi_dxj(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());
                } else {
                    val = dJi_dxi_dxj(x[k-1], Matrix::Zero(m,1), i, j) + dBh_dxi_dxj(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());
                }
                submtx(k,j) = val;
            }
            return dLL_dxi_dx;
        }


        // TODO obsolete
        Matrix dLLi_dxdx(const std::vector<Matrix>& x, const std::vector<Matrix>& u, const std::vector<Matrix>& h_plus_mask, const std::vector<Matrix>& lamda, const std::vector<Matrix>& mu, const int i) {
            throw std::runtime_error("obsolete function called");
            //cout << "dLLi_dxdx " << endl;
            Matrix dLL_dxdx = Matrix::Zero(T*N*n, T*N*n);

            auto submtx = [&](int k, int i, int j) {
                return dLL_dxdx.template block<n,n>((k - 1) * N * n + i * n, (k - 1) * N * n + j * n);
            };

            // dJi_dxi_dxi
            //cout << "dJi_dxi_dxi " << endl;
            for (int k = 1; k < T; ++k) {
                auto mtx = submtx(k, i, i);
                mtx = dJi_dxi_dxi(x[k - 1], u[k].row(i).transpose(), i);
                for (int j=0; j<N; ++j){
                    if (i==j){continue;}
                    if (h_plus_mask[k-1](i,j)){
                        mtx += mu[k - 1](i,j) * dh_dxi_dxi(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());
                    } else {
                        mtx += dBh_dxi_dxi(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());
                    }
                }
            }

            // dJi_dxi_dxi, k = T
            //cout << "dJi_dxi_dxi k=T " << endl;
            auto mtx = submtx(T, i, i);
            mtx = dJi_dxi_dxi(x[T - 1], Matrix::Zero(m,1), i);
            for (int j=0; j<N; ++j){
                if (i==j){continue;}
                if (h_plus_mask[T-1](i,j)){
                    mtx += mu[T - 1](i,j) * dh_dxi_dxi(x[T - 1].row(i).transpose(), x[T - 1].row(j).transpose());
                } else {
                    mtx += dBh_dxi_dxi(x[T - 1].row(i).transpose(), x[T - 1].row(j).transpose());
                }
            }

            // dJi_dxi_dxj
            //cout << "dJi_dxi_dxj " << endl;
            for (int k = 1; k < T; ++k) {
                for (int j = 0; j < N; ++j) {
                    if (i == j) {
                        continue;
                    }
                    Matrix val = Matrix::Zero(n, n);
                    if (h_plus_mask[k-1](i,j)){
                        val = dJi_dxi_dxj(x[k-1], u[k].row(i).transpose(), i, j) + mu[k - 1](i,j) * dh_dxi_dxj(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());
                        submtx(k,j,j) = dJi_dxj_dxj(x[k-1], u[k].row(i).transpose(), i, j) + mu[k - 1](i,j) * dh_dxj_dxj(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());
                    } else {
                        val = dJi_dxi_dxj(x[k-1], u[k].row(i).transpose(), i, j) + dBh_dxi_dxj(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());
                        submtx(k,j,j) = dJi_dxj_dxj(x[k-1], u[k].row(i).transpose(), i, j) + dBh_dxj_dxj(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());
                    }
                    submtx(k,i,j) = val;
                    submtx(k,j,i) = val.transpose();
                }
            }

            //cout << "dJi_dxi_dxj k=T" << endl;
            const int k = T;
            for (int j = 0; j < N; ++j) {
                if (i == j) {
                    continue;
                }
                Matrix val = Matrix::Zero(n, n);
                if (h_plus_mask[k-1](i,j)){
                    val = dJi_dxi_dxj(x[k-1], Matrix::Zero(m,1), i, j) + mu[k - 1](i,j) * dh_dxi_dxj(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());
                    submtx(k,j,j) = dJi_dxj_dxj(x[k-1], Matrix::Zero(m,1), i, j) + mu[k - 1](i,j) * dh_dxj_dxj(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());
                } else {
                    val = dJi_dxi_dxj(x[k-1], Matrix::Zero(m,1), i, j) + dBh_dxi_dxj(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());
                    submtx(k,j,j) = dJi_dxj_dxj(x[k-1], Matrix::Zero(m,1), i, j) + dBh_dxj_dxj(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());
                }
                submtx(k,i,j) = val;
                submtx(k,j,i) = val.transpose();
            }
            return dLL_dxdx;
        }

        Matrix dLLi_dxi_dmu(const std::vector<Matrix>& x, const std::vector<Matrix>& u, const std::vector<Matrix>& h_plus_mask, const std::vector<Matrix>& lamda, const std::vector<Matrix>& mu, int i) {
            // Calculate dimensions
            const int dim_x = T * N * n;
            const int dim_u = T * N * m;
            const int dim_mu = T * N * N;

            // Initialize dLL_dx_dmu matrix
            Matrix dLL_dxi_dmu = Matrix::Zero(T*n, dim_mu);

            for (int k = 1; k < T+1; ++k) {
                for (int j=0; j<N; j++) {
                    if (h_plus_mask[k-1](i,j)){
                        Matrix dLLi_dxki_dmuijk = dh_dxi(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());
                        dLL_dxi_dmu.template block<n,1>((k - 1) * n, (k - 1) * N * N + i * N + j) = dLLi_dxki_dmuijk.transpose();
                    }
                }
            }

            return dLL_dxi_dmu;
        }

        // TODO obsolete
        Matrix dLLi_dx_dmu(const std::vector<Matrix>& x, const std::vector<Matrix>& u, const std::vector<Matrix>& h_plus_mask, const std::vector<Matrix>& lamda, const std::vector<Matrix>& mu, int i) {
            throw std::runtime_error("obsolete function called");
            // Calculate dimensions
            const int dim_x = T * N * n;
            const int dim_u = T * N * m;
            const int dim_mu = T * N * N;

            // Initialize dLL_dx_dmu matrix
            Matrix dLL_dx_dmu = Matrix::Zero(dim_x, dim_mu);

            for (int k = 1; k < T+1; ++k) {
                for (int j=0; j<N; j++) {
                    if (h_plus_mask[k-1](i,j)){
                        Matrix dLLi_dxki_dmuijk = dh_dxi(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());
                        Matrix dLLi_dxkj_dmuijk = dh_dxj(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());

                        dLL_dx_dmu.template block<n,1>((k - 1) * N * n + i * n, (k - 1) * N * N + i * N + j) = dLLi_dxki_dmuijk.transpose();
                        dLL_dx_dmu.template block<n,1>((k - 1) * N * n + j * n, (k - 1) * N * N + i * N + j) = dLLi_dxkj_dmuijk.transpose();
                    }
                }
            }

            return dLL_dx_dmu;
        }


        // fill-in style
        template<typename Derived>
        void dr_dx(const std::vector<Matrix>& x, const std::vector<Matrix>& u, const std::vector<Matrix>& lamda, const std::vector<Matrix>& mu, const std::vector<Matrix>& h_plus_mask, const MatrixBase<Derived>& mtx) {
            int dim_x = N * T * n;
            int dim_u = N * T * m;
            auto& drdx = const_cast<MatrixBase<Derived>&>(mtx);
            // Initialize index
            int index = 0;

            for (int i = 0; i < N; ++i) {
                // Calculate dLL_dxdx
                Matrix dLL_dxi_dx = dLLi_dxi_dx(x, u, h_plus_mask, lamda, mu, i);
                drdx.block(index, 0, T * n, dim_x) = dLL_dxi_dx;
                index += T*n + T*m;

                Matrix dF0dx = dF0_dx(x, u, i);
                drdx.block(index, 0, n, dim_x) = dF0dx;

                for (int k = 1; k < T; ++k) {
                    Matrix dFdx = dF_dx(x, u, i, k);
                    drdx.block(index + n * k, 0, n, dim_x) = dFdx;
                }
                index += n*T;

                for (int k = 1; k <= T; ++k) {
                    Matrix dhdx = Matrix::Zero(h_plus_mask[k-1].row(i).count(), dim_x);
                    int dh_dx_idx = 0;
                    for (int j = 0; j < N; ++j) {
                        //if (i==j){continue;}
                        if (h_plus_mask[k-1](i,j)){
                            dhdx.row(dh_dx_idx) = dh_dx(x, k, i, j);
                            dh_dx_idx++;
                        }
                    }
                    // TODO we could set the original matrix directly to avoid copying
                    drdx.block(index, 0, dhdx.rows(), dim_x) = dhdx;
                    index += dhdx.rows();
                }
            }
        }

        // copy style
        Matrix dr_dx(const std::vector<Matrix>& x, const std::vector<Matrix>& u, const std::vector<Matrix>& lamda, const std::vector<Matrix>& mu, const std::vector<Matrix>& h_plus_mask) {
            // Calculate dimensions
            int dim_x = N * T * n;
            int dim_u = N * T * m;
            int h_plus_sum = 0;
            for (const auto& mask: h_plus_mask){
                h_plus_sum += mask.count();
            }
            int dim_r = N * (T*n + T*m + T * n) + h_plus_sum;

            // Initialize dr_dx matrix
            Matrix drdx = Matrix::Zero(dim_r, dim_x);
            dr_dx(x, u, lamda, mu, h_plus_mask, drdx);
            return drdx;
        }

        // fill a sparse matrix with dr_dx, starting at row/col_offset.
        void dr_dx_fill_block(const std::vector<Matrix>& x, const std::vector<Matrix>& u, const std::vector<Matrix>& lamda, const std::vector<Matrix>& mu, const std::vector<Matrix>& h_plus_mask, SpMatrix& mtx, const int row_offset, const int col_offset, const int row_size, const int col_size) {
            int dim_x = N * T * n;
            int dim_u = N * T * m;
            auto& drdx = mtx;
            // Initialize index
            int index = 0;

            for (int i = 0; i < N; ++i) {
                // Calculate dLL_dxdx
                Matrix dLL_dxi_dx = dLLi_dxi_dx(x, u, h_plus_mask, lamda, mu, i);
                //drdx.block(index, 0, T * n, dim_x) = dLL_dxi_dx;
                sp_assign(dLL_dxi_dx, drdx, index+row_offset, 0+col_offset, T * n, dim_x);
                index += T*n + T*m;

                Matrix dF0dx = dF0_dx(x, u, i);
                //drdx.block(index, 0, n, dim_x) = dF0dx;
                sp_assign(dF0dx,drdx, index+row_offset, 0+col_offset, n, dim_x);


                for (int k = 1; k < T; ++k) {
                    Matrix dFdx = dF_dx(x, u, i, k);
                    //drdx.block(index + n * k, 0, n, dim_x) = dFdx;
                    sp_assign(dFdx, drdx, index + n * k + row_offset, 0+col_offset, n, dim_x);
                }
                index += n*T;

                for (int k = 1; k <= T; ++k) {
                    Matrix dhdx = Matrix::Zero(h_plus_mask[k-1].row(i).count(), dim_x);
                    int dh_dx_idx = 0;
                    for (int j = 0; j < N; ++j) {
                        //if (i==j){continue;}
                        if (h_plus_mask[k-1](i,j)){
                            dhdx.row(dh_dx_idx) = dh_dx(x, k, i, j);
                            dh_dx_idx++;
                        }
                    }
                    // TODO we could set the original matrix directly to avoid copying
                    //drdx.block(index, 0, dhdx.rows(), dim_x) = dhdx;
                    sp_assign(dhdx, drdx, index+row_offset, 0+col_offset, dhdx.rows(), dim_x);
                    index += dhdx.rows();
                }
            }
        }


        template<typename Derived>
        void dr_du(const std::vector<Matrix>& x, const std::vector<Matrix>& u, const std::vector<Matrix>& lamda, const std::vector<Matrix>& mu, const std::vector<Matrix>& h_plus_mask, const MatrixBase<Derived>& mtx) {
            auto& drdu = const_cast<MatrixBase<Derived>&>(mtx);
            int index = 0;
            for (int i = 0; i < N; ++i) {
                index += T*n;
                int k = 0;
                Matrix dLL_duik_duik = dJi_dudu(x0,u[k].row(i).transpose(),i);
                drdu.template block<m,m>(index + k * m, k * N * m + i * m) = dLL_duik_duik;
                for (int k = 1; k < T; ++k) {
                    // dLL_duik_duik
                    Matrix dLL_duik_duik = dJi_dudu(x[k-1],u[k].row(i).transpose(),i);
                    drdu.template block<m,m>(index + k * m, k * N * m + i * m) = dLL_duik_duik;
                }
                index += T*m;

                drdu.template block<n,m>(index + 0 * n, 0 * N * m + i * m) = df_du(x0.row(i).transpose(), u[0].row(i).transpose(),i);
                for (int k = 1; k < T; ++k) {
                    drdu.template block<n,m>(index + k * n, k * N * m + i * m) = df_du(x[k - 1].row(i).transpose(), u[k].row(i).transpose(),i);
                }
                // skip count for h_plus_mask[all k, i, all j]
                int skip_count = 0;
                for (int k = 1; k < T+1; ++k) {
                    skip_count +=h_plus_mask[k-1].row(i).count();
                }
                index += n * T + skip_count;
            }
        }

        Matrix dr_du(const std::vector<Matrix>& x, const std::vector<Matrix>& u, const std::vector<Matrix>& lamda, const std::vector<Matrix>& mu, const std::vector<Matrix>& h_plus_mask) {
            // Calculate dimensions
            int dim_u = T * N * m;
            int h_plus_sum = 0;
            for (const auto& mask : h_plus_mask) {
                h_plus_sum += mask.count();
            }
            int dim_r = N * (T*n + T*m + T * n) + h_plus_sum;
            Matrix drdu = Matrix::Zero(dim_r, dim_u);
            dr_du(x, u, lamda, mu, h_plus_mask, drdu);
            return drdu;
        }

        void dr_du_fill_block(const std::vector<Matrix>& x, const std::vector<Matrix>& u, const std::vector<Matrix>& lamda, const std::vector<Matrix>& mu, const std::vector<Matrix>& h_plus_mask, SpMatrix& mtx, const int row_offset, const int col_offset, const int row_size, const int col_size) {
            auto& drdu = mtx;
            int index = 0;
            for (int i = 0; i < N; ++i) {
                index += T*n;
                int k = 0;
                Matrix dLL_duik_duik = dJi_dudu(x0,u[k].row(i).transpose(),i);
                //drdu.template block<m,m>(index + k * m, k * N * m + i * m) = dLL_duik_duik;
                sp_assign(dLL_duik_duik, drdu, index + k * m + row_offset, k * N * m + i * m + col_offset, m, m);
                for (int k = 1; k < T; ++k) {
                    // dLL_duik_duik
                    Matrix dLL_duik_duik = dJi_dudu(x[k-1],u[k].row(i).transpose(),i);
                    //drdu.template block<m,m>(index + k * m, k * N * m + i * m) = dLL_duik_duik;
                    sp_assign(dLL_duik_duik, drdu,index + k * m+row_offset, k * N * m + i * m +col_offset, m, m);
                }
                index += T*m;

                //drdu.template block<n,m>(index + 0 * n, 0 * N * m + i * m) = df_du(x0.row(i).transpose(), u[0].row(i).transpose(),i);
                sp_assign(df_du(x0.row(i).transpose(), u[0].row(i).transpose(),i), drdu, index + 0 * n + row_offset, 0 * N * m + i * m + col_offset, n, m);
                for (int k = 1; k < T; ++k) {
                    //drdu.template block<n,m>(index + k * n, k * N * m + i * m) = df_du(x[k - 1].row(i).transpose(), u[k].row(i).transpose(),i);
                    sp_assign(df_du(x[k - 1].row(i).transpose(), u[k].row(i).transpose(),i), drdu, index + k * n + row_offset, k * N * m + i * m + col_offset, n, m);
                }
                // skip count for h_plus_mask[all k, i, all j]
                int skip_count = 0;
                for (int k = 1; k < T+1; ++k) {
                    skip_count +=h_plus_mask[k-1].row(i).count();
                }
                index += n * T + skip_count;
            }
        }


        template<typename Derived>
        void dr_dlamda(const std::vector<Matrix>& x, const std::vector<Matrix>& u, const std::vector<Matrix>& lamda, const std::vector<Matrix>& mu, const std::vector<Matrix>& h_plus_mask, const MatrixBase<Derived>& mtx) {
            auto& drdlamda = const_cast<MatrixBase<Derived>&>(mtx);
            int index = 0;
            for (int i = 0; i < N; ++i) {
                for (int k = 1; k < T; ++k) {
                    // dLLi_dxki_dlamda_ki
                    drdlamda.template block<n,n>(index + (k - 1) * n, k * N * n + i * n) = df_dx(x[k - 1].row(i).transpose(), u[k].row(i).transpose(),i).transpose();
                    // dLLi_dxki_dlamda_k-1,i
                    drdlamda.template block<n,n>(index + (k - 1) * n, (k - 1) * N * n + i * n) = -Matrix::Identity(n, n);
                }

                const int k = T;
                drdlamda.template block<n,n>(index + (k - 1) * n, (k - 1) * N * n + i * n) = -Matrix::Identity(n, n);
                // skip dLL_dx, index now points at dLLi_du
                index += T * n;

                drdlamda.template block<m,n>(index + 0 * m, 0 * N * n + i * n) = df_du(x0.row(i).transpose(), u[0].row(i).transpose(),i).transpose();
                for (int k = 1; k < T; ++k) {
                    drdlamda.template block<m,n>(index + k * m, k * N * n + i * n) = df_du(x[k-1].row(i).transpose(), u[k].row(i).transpose(),i).transpose();
                }

                // skip count for h_plus_mask[all k, i, all j]
                int skip_count = 0;
                for (int k = 1; k < T+1; ++k) {
                    skip_count +=h_plus_mask[k-1].row(i).count();
                }
                index += T * m + n * T + skip_count;
            }

        }

        Matrix dr_dlamda(const std::vector<Matrix>& x, const std::vector<Matrix>& u, const std::vector<Matrix>& lamda, const std::vector<Matrix>& mu, const std::vector<Matrix>& h_plus_mask) {
            // Calculate dimensions
            int dim_lamda = T * N * n;
            int h_plus_sum = 0;
            for (const auto& mask : h_plus_mask) {
                h_plus_sum += mask.count();
            }
            int dim_r = N * (T*n + T*m + T * n) + h_plus_sum;
            Matrix drdlamda = Matrix::Zero(dim_r, dim_lamda);
            dr_dlamda(x, u, lamda, mu, h_plus_mask, drdlamda);
            return drdlamda;
        }

        void dr_dlamda_fill_block(const std::vector<Matrix>& x, const std::vector<Matrix>& u, const std::vector<Matrix>& lamda, const std::vector<Matrix>& mu, const std::vector<Matrix>& h_plus_mask, SpMatrix& mtx, const int row_offset, const int col_offset, const int row_size, const int col_size) {
            auto& drdlamda = mtx;
            int index = 0;
            for (int i = 0; i < N; ++i) {
                for (int k = 1; k < T; ++k) {
                    // dLLi_dxki_dlamda_ki
                    //drdlamda.template block<n,n>(index + (k - 1) * n, k * N * n + i * n) = df_dx(x[k - 1].row(i).transpose(), u[k].row(i).transpose(),i).transpose();
                    sp_assign(df_dx(x[k - 1].row(i).transpose(), u[k].row(i).transpose(),i).transpose(), drdlamda, index + (k - 1) * n + row_offset, k * N * n + i * n + col_offset, n, n);
                    // dLLi_dxki_dlamda_k-1,i
                    //drdlamda.template block<n,n>(index + (k - 1) * n, (k - 1) * N * n + i * n) = -Matrix::Identity(n, n);
                    sp_assign(-Matrix::Identity(n, n), drdlamda, index + (k - 1) * n + row_offset, (k - 1) * N * n + i * n + col_offset, n, n);
                }

                const int k = T;
                //drdlamda.template block<n,n>(index + (k - 1) * n, (k - 1) * N * n + i * n) = -Matrix::Identity(n, n);
                sp_assign(-Matrix::Identity(n, n), drdlamda, index + (k - 1) * n + row_offset, (k - 1) * N * n + i * n + col_offset, n, n);
                // skip dLL_dx, index now points at dLLi_du
                index += T * n;

                //drdlamda.template block<m,n>(index + 0 * m, 0 * N * n + i * n) = df_du(x0.row(i).transpose(), u[0].row(i).transpose(),i).transpose();
                sp_assign(df_du(x0.row(i).transpose(), u[0].row(i).transpose(),i).transpose(), drdlamda, index + 0 * m + row_offset, 0 * N * n + i * n + col_offset, m, n);
                for (int k = 1; k < T; ++k) {
                    //drdlamda.template block<m,n>(index + k * m, k * N * n + i * n) = df_du(x[k-1].row(i).transpose(), u[k].row(i).transpose(),i).transpose();
                    sp_assign(df_du(x[k-1].row(i).transpose(), u[k].row(i).transpose(),i).transpose(), drdlamda, index + k * m + row_offset, k * N * n + i * n + col_offset, m, n);
                }

                // skip count for h_plus_mask[all k, i, all j]
                int skip_count = 0;
                for (int k = 1; k < T+1; ++k) {
                    skip_count +=h_plus_mask[k-1].row(i).count();
                }
                index += T * m + n * T + skip_count;
            }

        }

        template<typename Derived>
        void dr_dmu(const std::vector<Matrix>& x, const std::vector<Matrix>& u, const std::vector<Matrix>& lamda, const std::vector<Matrix>& mu, const std::vector<Matrix>& h_plus_mask, const MatrixBase<Derived>& mtx) {
            auto& drdmu = const_cast<MatrixBase<Derived>&>(mtx);
            const int dim_mu = T * N * N;
            int index = 0;
            for (int i = 0; i < N; ++i) {
                // Calculate dLL_dx_dmu
                Matrix dLL_dxi_dmu = dLLi_dxi_dmu(x, u, h_plus_mask, lamda, mu, i);
                drdmu.block(index, 0, T*n, dim_mu) = dLL_dxi_dmu;
                // skip count for h_plus_mask[all k, i, all j]
                int skip_count = 0;
                for (int k = 1; k < T+1; ++k) {
                    skip_count +=h_plus_mask[k-1].row(i).count();
                }
                index += T*n + T*m + n * T + skip_count;
            }

        }

        Matrix dr_dmu(const std::vector<Matrix>& x, const std::vector<Matrix>& u, const std::vector<Matrix>& lamda, const std::vector<Matrix>& mu, const std::vector<Matrix>& h_plus_mask) {
            // Calculate dimensions
            const int dim_mu = T * N * N;
            int h_plus_sum = 0;
            for (const auto& mask : h_plus_mask) {
                h_plus_sum += mask.count();
            }
            int dim_r = N * (T*n + T*m + T * n) + h_plus_sum;

            // Initialize dr_dmu matrix
            Matrix drdmu = Matrix::Zero(dim_r, dim_mu);
            dr_dmu(x, u, lamda, mu, h_plus_mask, drdmu);
            return drdmu;
        }

        void dr_dmu_fill_block(const std::vector<Matrix>& x, const std::vector<Matrix>& u, const std::vector<Matrix>& lamda, const std::vector<Matrix>& mu, const std::vector<Matrix>& h_plus_mask, SpMatrix& mtx, const int row_offset, const int col_offset, const int row_size, const int col_size) {
            auto& drdmu = mtx;
            const int dim_mu = T * N * N;
            int index = 0;
            for (int i = 0; i < N; ++i) {
                // Calculate dLL_dx_dmu
                Matrix dLL_dxi_dmu = dLLi_dxi_dmu(x, u, h_plus_mask, lamda, mu, i);
                //drdmu.block(index, 0, T*n, dim_mu) = dLL_dxi_dmu;
                sp_assign(dLL_dxi_dmu, drdmu, index + row_offset, 0+col_offset, T*n, dim_mu);
                // skip count for h_plus_mask[all k, i, all j]
                int skip_count = 0;
                for (int k = 1; k < T+1; ++k) {
                    skip_count +=h_plus_mask[k-1].row(i).count();
                }
                index += T*n + T*m + n * T + skip_count;
            }

        }

        std::vector<Matrix> getHplusMask(const std::vector<Matrix>& x) {
             std::vector<Matrix> h_plus_mask(T);
             for (int i=0; i<T; i++){
                 h_plus_mask[i]= Matrix::Zero(N, N);
             }

            for (int k = 1; k < T+1; ++k) {
                for (int i = 0; i < N; ++i) {
                    for (int j = i + 1; j < N; ++j) {
                        h_plus_mask[k-1](i, j) = h_plus_mask[k-1](j, i) = h(x[k-1].row(i).transpose(), x[k-1].row(j).transpose()) >= 0;
                    }
                }
            }
            return h_plus_mask;
        }

        Scalar getCollisionResidual(const std::vector<Matrix>& x) {
            Scalar h_res = 0;
            for (int k = 1; k < T+1; ++k) {
                for (int i = 0; i < N; ++i) {
                    for (int j = i + 1; j < N; ++j) {
                        Scalar this_h = h(x[k-1].row(i).transpose(), x[k-1].row(j).transpose());
                        if (this_h > 0){
                            h_res += this_h;
                        }
                    }
                }
            }
            return h_res;
        }

        Matrix r(const std::vector<Matrix>& x, const std::vector<Matrix>& u, const std::vector<Matrix>& lamda, const std::vector<Matrix>& mu, const std::vector<Matrix>& h_plus_mask) {
            int h_plus_sum = 0;
            for (const auto& mask : h_plus_mask) {
                h_plus_sum += mask.count();
            }
            const int dim_r = N * (T*n + T*m + T * n) + h_plus_sum;

            Matrix r = Matrix::Zero(dim_r,1);
            int index = 0;
            for (int i = 0; i < N; ++i) {
                Matrix dLL_dxi = dLLi_dxi(x, u, h_plus_mask, lamda, mu, i).transpose();
                Matrix dLL_dui = dLLi_dui(x, u, h_plus_mask, lamda, mu, i).transpose();
                r.block(index, 0, T*n, 1) = dLL_dxi;
                index += T*n;
                r.block(index, 0, T*m, 1) = dLL_dui;
                index += T*m;

                // Dynamics for f(x0,u0) = x1
                r.template block<n,1>(index, 0) = f(x0.row(i).transpose(), u[0].row(i).transpose()) - x[0].row(i).transpose();

                for (int k = 1; k < T; ++k) {
                    r.template block<n,1>(index+k*n,0) = f(x[k - 1].row(i).transpose(), u[k].row(i).transpose()) - x[k].row(i).transpose();
                }
                index += n * T;

                for (int k = 1; k < T+1; ++k) {
                    int h_idx = 0;
                    for (int j=0; j<N; ++j){
                        if (h_plus_mask[k-1](i,j)){
                            r(index,0) = h(x[k - 1].row(i).transpose(), x[k - 1].row(j).transpose());
                            h_idx++;
                        }
                    }
                    index += h_idx;
                }
            }
            return r;
        }

        Matrix dr_dy(const std::vector<Matrix>& x, const std::vector<Matrix>& u, const std::vector<Matrix>& lamda, const std::vector<Matrix>& mu, const std::vector<Matrix>& h_plus_mask) {
            int h_plus_sum = 0;
            for (const auto& mask : h_plus_mask) {
                h_plus_sum += mask.count();
            }
            const int dim_x = N * T * n;
            const int dim_u = N * T * m;
            const int dim_lamda = T * N * n;
            const int dim_mu = T * N * N;
            const int dim_r = N * (T*n + T*m + T * n) + h_plus_sum;

            const int dim_y = dim_x + dim_u + dim_lamda + dim_mu;
            Matrix Dr(dim_r,dim_y);
            Dr.setZero();

            //cout << "drdx: " << endl;
            dr_dx(x, u, lamda, mu, h_plus_mask, Dr.block(0,0,dim_r,dim_x));
            //cout << "drdu: " << endl;
            dr_du(x, u, lamda, mu, h_plus_mask, Dr.block(0,dim_x,dim_r,dim_u));
            //cout << "drdlamda: " << endl;
            dr_dlamda(x, u, lamda, mu, h_plus_mask, Dr.block(0,dim_x+dim_u,dim_r,dim_lamda));
            //cout << "drdmu: " << endl;
            dr_dmu(x, u, lamda, mu, h_plus_mask, Dr.block(0,dim_x+dim_u+dim_lamda,dim_r,dim_mu));
            return Dr;
        }

        SpMatrix dr_dy_sparse(const std::vector<Matrix>& x, const std::vector<Matrix>& u, const std::vector<Matrix>& lamda, const std::vector<Matrix>& mu, const std::vector<Matrix>& h_plus_mask) {
            int h_plus_sum = 0;
            for (const auto& mask : h_plus_mask) {
                h_plus_sum += mask.count();
            }
            const int dim_x = N * T * n;
            const int dim_u = N * T * m;
            const int dim_lamda = T * N * n;
            const int dim_mu = T * N * N;
            const int dim_r = N * (T*n + T*m + T * n) + h_plus_sum;

            const int dim_y = dim_x + dim_u + dim_lamda + dim_mu;
            SpMatrix Dr(dim_r,dim_y);
            //Dr.reserve(int(dim_y*dim_y*0.01));
            Dr.reserve(Eigen::VectorXi::Constant(dim_y,int(dim_r*0.01)));
            //Dr.setZero();

            //cout << "drdx: " << endl;
            dr_dx_fill_block(x, u, lamda, mu, h_plus_mask, Dr,0,0,dim_r,dim_x);
            //cout << "drdu: " << endl;
            dr_du_fill_block(x, u, lamda, mu, h_plus_mask, Dr,0,dim_x,dim_r,dim_u);
            //cout << "drdlamda: " << endl;
            dr_dlamda_fill_block(x, u, lamda, mu, h_plus_mask, Dr,0,dim_x+dim_u,dim_r,dim_lamda);
            //cout << "drdmu: " << endl;
            dr_dmu_fill_block(x, u, lamda, mu, h_plus_mask, Dr,0,dim_x+dim_u+dim_lamda,dim_r,dim_mu);
            return Dr;
        }

        std::vector<std::vector<Matrix>> step(const std::vector<Matrix>& x, const std::vector<Matrix>& u, const std::vector<Matrix>& lamda, const std::vector<Matrix>& mu) {
            //int additional_memory_usage_kb = getCurrentMemoryUsageInKB() - current_memory_usage_kb;
            //std::cout << "step entry memory: " << additional_memory_usage_kb << "KB" << std::endl;

            //cout << "step()" << endl;
            const auto h_plus_mask = getHplusMask(x);
            int h_plus_sum = 0;
            for (const auto& mask : h_plus_mask) {
                h_plus_sum += mask.count();
            }
            //cout << "getHplusMask()" << endl;
            profiler.s();

            const int dim_x = N * T * n;
            const int dim_u = N * T * m;
            const int dim_lamda = T * N * n;
            const int dim_mu = T * N * N;
            const int dim_r = N * (T*n + T*m + T * n) + h_plus_sum;
            const int dim_y = dim_x + dim_u + dim_lamda + dim_mu;

            //cout << "r()" << endl;
            auto r0 = r(x, u, lamda, mu, h_plus_mask);
            /*
            profiler.s("build dense");
            const Matrix Dr_dense = dr_dy(x, u, lamda, mu, h_plus_mask);
            profiler.e("build dense");
            // remove zero rows/cols
            profiler.s("dense nonzeros");
            std::vector<int> nonzero_cols_idx_dense;
            nonzero_cols_idx_dense = nonzero_cols(Dr_dense);
            Matrix Dr_reduced_dense = Dr_dense(Eigen::all, nonzero_cols_idx_dense);
            SpMatrix Dr_reduced_sparse = Dr_reduced_dense.sparseView();
            profiler.e("dense nonzeros");
            */


            profiler.s("build sparse");
            SpMatrix Dr_sparse = dr_dy_sparse(x, u, lamda, mu, h_plus_mask);
            profiler.e("build sparse");
            // NOTE here the memory occupied by Dr is not released
            profiler.s("sparse nonzeros");
            std::vector<int> nonzero_cols_idx;
            SpMatrix Dr_reduced;
            std::tie(Dr_reduced, nonzero_cols_idx) = remove_empty_cols(Dr_sparse,dim_r);
            profiler.e("sparse nonzeros");

            profiler.s("solve");
            Eigen::LeastSquaresConjugateGradient<SpMatrix> solver;
            solver.setTolerance(1e-5);
            // solve r0 + Dr* dy = 0 least square
            solver.compute(Dr_reduced);
            //solver.compute(Dr.sparseView());
            //cout << "compute" << endl;

            if (solver.info() != Eigen::Success){
                throw std::runtime_error(" solver initialization failed");
                //return std::vector<std::vector<Matrix>>();
            }

            Matrix dy_reduced = solver.solve(-r0);
            if (solver.info() != Eigen::Success){
                throw std::runtime_error(" solver solve() failed");
                //return std::vector<std::vector<Matrix>>();
            }
            profiler.e("solve");

            // reconstruct dy from dy_reduced
            profiler.s("reconstruct dy");
            Matrix dy(dim_y,1);
            dy.setZero();
            dy(nonzero_cols_idx,Eigen::all) = dy_reduced;
            profiler.e("reconstruct dy");

            // line search
            profiler.s("line search");
            Scalar step = 1.0; // step size
            Scalar r0_norm = r0.norm();
            Scalar rt_norm = r0_norm;
            Scalar apriori_h_res = getCollisionResidual(x);

            auto split_y = [&](const std::vector<Matrix>& x, const std::vector<Matrix>& u, const std::vector<Matrix>& lamda, const std::vector<Matrix>& mu, const Matrix& dy, Scalar my_step){
                const auto x_size = x.size();
                std::vector<Matrix> xx(x_size);
                for (int i=0; i<x_size; i++){
                    xx.at(i) = x.at(i) + my_step * dy.block(i*N*n,0,N*n,1).reshaped<Eigen::RowMajor>(N,n);
                }
                const auto u_size = u.size();
                std::vector<Matrix> uu(x_size);
                for (int i=0; i<u_size; i++){
                    uu.at(i) = u.at(i) + my_step * dy.block(dim_x+i*N*m,0,N*m,1).reshaped<Eigen::RowMajor>(N,m);
                }
                const auto lamda_size = lamda.size();
                std::vector<Matrix> ll(lamda_size);
                for (int i=0; i<lamda_size; i++){
                    ll.at(i) = lamda.at(i) + my_step * dy.block(dim_x+dim_u+i*N*n,0,N*n,1).reshaped<Eigen::RowMajor>(N,n);
                }
                const auto mu_size = mu.size();
                std::vector<Matrix> mm(mu_size);
                for (int i=0; i<mu_size; i++){
                    mm.at(i) = mu.at(i) + my_step * dy.block(dim_x+dim_u+dim_lamda+i*N*N,0,N*N,1).reshaped<Eigen::RowMajor>(N,N);
                }
                return std::tuple<std::vector<Matrix>,std::vector<Matrix>,std::vector<Matrix>,std::vector<Matrix>> {xx, uu, ll, mm};
            };


            auto r_t_norm = [&](Scalar my_step){
                auto y_tuple = split_y(x, u, lamda, mu, dy, my_step);
                return r(std::get<0>(y_tuple), std::get<1>(y_tuple), std::get<2>(y_tuple),std::get<3>(y_tuple), h_plus_mask).norm();
            };


            bool flag_no_step = true;
            for (int i=0; i<backtracking_max_iter; i++){
                auto y_tuple = split_y(x, u, lamda, mu, dy, step);
                rt_norm = r(std::get<0>(y_tuple), std::get<1>(y_tuple), std::get<2>(y_tuple),std::get<3>(y_tuple), h_plus_mask).norm();
                if (rt_norm > (1-bc_a*step)*r0_norm){
                    step *= bc_b;
                } else {
                    Scalar search_h_res = getCollisionResidual(std::get<0>(y_tuple));
                    if (search_h_res > apriori_h_res){
                        step *= bc_b;
                    } else {
                        flag_no_step = false;
                        break;
                    }
                }
            }
            profiler.e("line search");
            profiler.e();

            //additional_memory_usage_kb = getCurrentMemoryUsageInKB() - current_memory_usage_kb;
            //std::cout << "step exit memory: " << additional_memory_usage_kb << "KB" << std::endl;

            // stopping criteria
            auto y_tuple = split_y(x, u, lamda, mu, dy, step);
            if ( abs(rt_norm) < tolerance and h_plus_sum == 0){
                // stopping
                throw pybind11::stop_iteration("stopping criteria met");
                //return std::vector<std::vector<Matrix>>();
            } else if ( flag_no_step){
                throw pybind11::stop_iteration("iteration not making progress");
            } else {
                auto xx = std::get<0>(y_tuple);
                auto uu = std::get<1>(y_tuple);
                auto ll = std::get<2>(y_tuple);
                auto mm = std::get<3>(y_tuple);
                return std::vector<std::vector<Matrix>>{xx,uu,ll,mm};
            }
        }


        // --- helper function ---
        // find nonzero submatrix
        // return: skimmed matrix (dense), nonzero row indices, nonzero col indices
        std::vector<int>
        nonzero_cols(const Matrix& mtx){
            // cols
            Eigen::Matrix<bool,1,Eigen::Dynamic,Eigen::RowMajor> nonzero_cols_mask = mtx.cast<bool>().colwise().any();
            const int nonzero_cols_size = nonzero_cols_mask.cast<int>().sum();
            std::vector<int> nonzero_cols_idx;
            nonzero_cols_idx.reserve(nonzero_cols_size);
            const int mtx_cols = mtx.cols();
            for (int i=0; i<mtx_cols; i++){
                if (nonzero_cols_mask(0,i)){
                nonzero_cols_idx.push_back(i);
                }
            }

            return  nonzero_cols_idx;
        }

        void summary(){
            profiler.summary();
        }

        // DEBUG functions

        // solve Ax=B
        Matrix SparseQR(const Matrix& A, const Matrix& B){
            Eigen::SparseQR<SpMatrix, Eigen::COLAMDOrdering<int>> solver;
            solver.compute(A.sparseView());
            if (solver.info() != Eigen::Success){
                throw std::runtime_error(" solver initialization failed");
                return Matrix{};
            }

            Matrix x = solver.solve(B);
            if (solver.info() != Eigen::Success){
                throw std::runtime_error(" solver solve() failed");
                return Matrix{};
            }
            return x;
        }

        Matrix LeastSquaresConjugateGradient(const Matrix& A, const Matrix& B){
            Eigen::LeastSquaresConjugateGradient<SpMatrix> solver;
            solver.compute(A.sparseView());
            if (solver.info() != Eigen::Success){
                throw std::runtime_error(" solver initialization failed");
                return Matrix{};
            }

            // solver.setMaxIterations();
            // solver.setTolerance
            Matrix x = solver.solve(B);
            if (solver.info() != Eigen::Success){
                throw std::runtime_error(" solver solve() failed");
                return Matrix{};
            }
            return x;
        }

        void print_dim(const Matrix val){
            std::cout << "rows " << val.rows() << "cols " << val.cols() << endl;
        }
        Matrix test_bool_array(const Matrix val, const Matrix mask){
            Matrix output(val);
            for (int i=0; i<val.rows(); i++){
                for (int j=0; j<val.cols(); j++){
                    if (!mask(i,j)){
                        output(i,j) = 0;
                    }
                }
            }
            return output;
        }
        Matrix three_dim(const std::vector<Matrix> mtx_vec){
            return mtx_vec[1];
        }
        // doesn't work unfortunately
        void pass_by_ref(std::vector<Matrix>& array){
            // multiply the first array value by 2
            array[0] *= 2;
            // multiply the first array value by 0.5
            array[1] *= 0.5;
            return;
        }

        // ---- virtual functions, they should be overridden in derived class
        virtual Matrix f(const Matrix x, const Matrix u){
            throw std::runtime_error("abstract function f() shouldn't be called");
            return x;
        }
        virtual Matrix df_dx(const Matrix x, const Matrix u, const int i){
            throw std::runtime_error("abstract function df_dx() shouldn't be called");
            return x;
        }
        virtual Matrix df_du(const Matrix x, const Matrix u, const int i){
            throw std::runtime_error("abstract function hdf_du() shouldn't be called");
            return x;
        }


        // TODO use correct dimension zero matrices
        // collision constraint function
        virtual Scalar h(const Matrix x_i, const Matrix x_j){
            throw std::runtime_error("abstract function h() shouldn't be called");
            return 0.0;
        }
        virtual Matrix dh_dxi(const Matrix x_i, const Matrix x_j){
            throw std::runtime_error("abstract function dh_dxi() shouldn't be called");
            return x_i;
        }
        virtual Matrix dh_dxj(const Matrix x_i, const Matrix x_j){
            throw std::runtime_error("abstract function dh_dxj() shouldn't be called");
            return x_i;
        }
        virtual Matrix dh_dxi_dxi(const Matrix x_i, const Matrix x_j){
            throw std::runtime_error("abstract function dh_dxi_dxi() shouldn't be called");
            return x_i;
        }
        virtual Matrix dh_dxj_dxi(const Matrix x_i, const Matrix x_j){
            throw std::runtime_error("abstract function dh_dxj_dxi() shouldn't be called");
            return x_i;
        }
        virtual Matrix dh_dxi_dxj(const Matrix x_i, const Matrix x_j){
            throw std::runtime_error("abstract function dh_dxi_dxj() shouldn't be called");
            return x_i;
        }
        virtual Matrix dh_dxj_dxj(const Matrix x_i, const Matrix x_j){
            throw std::runtime_error("abstract function dh_dxj_dxj() shouldn't be called");
            return x_i;
        }

        // Objective function (J)
        virtual Matrix J(const Matrix x_k, const Matrix u_k_i, int i){
            throw std::runtime_error("abstract function J() shouldn't be called");
            return x_k;
        }
        virtual Matrix dJi_dxi(const Matrix x_k, const Matrix u, int i){
            throw std::runtime_error("abstract function dJi_dxi() shouldn't be called");
            return x_k;
        }
        virtual Matrix dJi_dxj(const Matrix x_k, const Matrix u, int i, int j){
            throw std::runtime_error("abstract function dJi_dxj() shouldn't be called");
            return x_k;
        }
        virtual Matrix dJi_du(const Matrix x_k, const Matrix u, int i){
            throw std::runtime_error("abstract function dJi_du() shouldn't be called");
            return x_k;
        }
        virtual Matrix dJi_dxi_dxi(const Matrix x_k, const Matrix u, int i){
            throw std::runtime_error("abstract function dJi_dxi_dxi() shouldn't be called");
            return x_k;
        }
        virtual Matrix dJi_dxi_dxj(const Matrix x_k, const Matrix u, int i, int j){
            throw std::runtime_error("abstract function dJi_dxi_dxj() shouldn't be called");
            return x_k;
        }
        virtual Matrix dJi_dxj_dxj(const Matrix x_k, const Matrix u, int i,int j){
            throw std::runtime_error("abstract function dJi_dxj_dxj() shouldn't be called");
            return x_k;
        }
        virtual Matrix dJi_dudu(const Matrix x_k, const Matrix u, int i){
            throw std::runtime_error("abstract function dJi_dudu() shouldn't be called");
            return x_k;
        }
};
