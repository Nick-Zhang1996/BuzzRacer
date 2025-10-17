#include "sparse_residual_game.h"
constexpr int n = 4;
constexpr int m = 2;

class UnstructuredDriving : public ResidualGame<n,m> {
    private:
        Matrix A,B,J_Qr,J_Q,J_R,h_Qh,target_y;
    public:
        UnstructuredDriving(const int _N, const int _T,
                const Scalar _dt, const Scalar _rho, const Scalar _rho_b, const Scalar _bc_a, const Scalar _bc_b,
                const Scalar _tolerance, const int _backtracking_max_iter,
                const Matrix _J_Qr, const Matrix _J_Q, const Matrix _J_R, const Matrix _A, const Matrix _B, const Matrix _h_Qh, const Matrix _target_y):
            ResidualGame(_N, _T, _dt, _rho, _rho_b, _bc_a, _bc_b, _tolerance, _backtracking_max_iter), J_Qr(_J_Qr), J_Q(_J_Q), J_R(_J_R), A(_A), B(_B), h_Qh(_h_Qh), target_y(_target_y) {
                /*
                cout << "N = " << N;
                cout << "T = " << T;
                cout << "n = " << n;
                cout << "m = " << m;
                */
        }

        // TODO unnecessary copy
        // TODO fixed dimension arrays
        void set_A(const Matrix &val){
            A = Matrix(val);
        }
        void set_B(const Matrix &val){
            B = Matrix(val);
        }

        Matrix f(const Matrix x, const Matrix u){
            return A * x + B * u;
        }
        Matrix df_dx(const Matrix x, const Matrix u, const int i){
            return A;
        }
        Matrix df_du(const Matrix x, const Matrix u, const int i){
            return B;
        }

        // collision constraint function
        Scalar h(const Matrix x_i, const Matrix x_j){
            return -sqr(x_i(0,0)-x_j(0,0)) - sqr(x_i(1,0)-x_j(1,0)) + sqr(1.2);
        }
        Matrix dh_dxi(const Matrix x_i, const Matrix x_j){
            return 2*(x_i-x_j).transpose() * h_Qh;
        }
        Matrix dh_dxj(const Matrix x_i, const Matrix x_j){
            return 2*(x_j-x_i).transpose() * h_Qh;
        }
        Matrix dh_dxi_dxi(const Matrix x_i, const Matrix x_j){
            return 2*h_Qh.transpose();
        }
        Matrix dh_dxj_dxi(const Matrix x_i, const Matrix x_j){
            return -2*h_Qh.transpose();
        }
        Matrix dh_dxi_dxj(const Matrix x_i, const Matrix x_j){
            return -2*h_Qh.transpose();
        }
        Matrix dh_dxj_dxj(const Matrix x_i, const Matrix x_j){
            return 2*h_Qh.transpose();
        }

        // Objective function (J)
        // NOTE this is dependent upon the car
        Matrix J_x_ref_fun(int i){
            Matrix mtx(n,1);
            (mtx << 0,target_y(i,0), 2.0, 0.0 ).finished();
            return mtx;
        }
        Matrix J(const Matrix x_k, const Matrix u_k_i, int i){
            return (x_k.row(i).transpose()-J_x_ref_fun(i)).transpose() * J_Qr * (x_k.row(i).transpose()-J_x_ref_fun(i)) + x_k.row(i) * J_Q * x_k.row(i).transpose() + u_k_i.transpose() * J_R * u_k_i;
        }
        Matrix dJi_dxi(const Matrix x_k, const Matrix u, int i){
            return  2* (x_k.row(i).transpose()-J_x_ref_fun(i)).transpose() * J_Qr + 2*x_k.row(i).transpose().transpose() * J_Q;
        }
        Matrix dJi_dxj(const Matrix x_k, const Matrix u, int i, int j){
            return  Matrix::Zero(1,n);
        }
        Matrix dJi_du(const Matrix x_k, const Matrix u, int i){
            return  2* u.transpose() * J_R;
        }
        Matrix dJi_dxi_dxi(const Matrix x_k, const Matrix u, int i){
            return  2*J_Qr + 2*J_Q;
        }
        Matrix dJi_dxi_dxj(const Matrix x_k, const Matrix u, int i, int j){
            return  Matrix::Zero(n,n);
        }
        Matrix dJi_dxj_dxj(const Matrix x_k, const Matrix u, int i, int j){
            return  Matrix::Zero(n,n);
        }
        Matrix dJi_dudu(const Matrix x_k, const Matrix u, int i){
            return  2*J_R;
        }
};
