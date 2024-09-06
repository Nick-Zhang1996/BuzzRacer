#include "sparse_residual_game.h"
#include <math.h>

constexpr int n = 4;
constexpr int m = 2;

class CarRacing : public ResidualGame<n,m> {
    private:
        Matrix A,B,J_Qr,J_Q,J_R,h_Qh,target_y;
        std::vector<std::tuple<Scalar,Scalar>> curvature_vec;
        Scalar car_size;
    public:
        CarRacing(const int _N, const int _T,
                const Scalar _dt, const Scalar _rho, const Scalar _rho_b, const Scalar _bc_a, const Scalar _bc_b,
                const Scalar _tolerance, const int _backtracking_max_iter,
                const Matrix _J_Qr, const Matrix _J_Q, const Matrix _J_R, const Matrix _h_Qh, const Matrix _target_y):
            ResidualGame(_N, _T, _dt, _rho, _rho_b, _bc_a, _bc_b, _tolerance, _backtracking_max_iter),J_Qr(_J_Qr), J_Q(_J_Q), J_R(_J_R), h_Qh(_h_Qh), target_y(_target_y),car_size(0.2) {
                /*
                cout << "N = " << N;
                cout << "T = " << T;
                cout << "n = " << n;
                cout << "m = " << m;
                */
        }

        // a list of sorted pairs (distance, curvature) for the racetrack
        void set_curvature_vector(const std::vector<std::tuple<Scalar,Scalar>> in_curvature_vec){
            curvature_vec = std::move(in_curvature_vec);
        }

        Scalar curvature_fun(Scalar s){
            s = fmod(s, std::get<0>(curvature_vec.back()) );
            auto it = std::lower_bound(curvature_vec.begin(), curvature_vec.end(), s,
                    [](const std::tuple<Scalar, Scalar>& val, Scalar key){
                        return std::get<0>(val) < key;
                    });
            auto it_prev = it;
            if (it == curvature_vec.begin()){
                it_prev = curvature_vec.end()-1;
            } else {
                it_prev = it - 1;
            }

            float x1 = std::get<0>(*it_prev);
            float y1 = std::get<1>(*it_prev);
            float x2 = std::get<0>(*it);
            float y2 = std::get<1>(*it);

            return y1 + (y2 - y1) * (s - x1) / (x2 - x1);
        }


        Matrix f(const Matrix x, const Matrix u, const int i){
            const Scalar lf = 1.0; const Scalar lr = 1.0;
            const Scalar beta = atan(tan(u(1,0))*lr/(lf+lr));
            Scalar k_s = curvature_fun(x(0,0));
            Matrix dx = (Matrix(n,1) <<
                x(1,0)*cos(x(3,0))/(1-x(2,0)*k_s),
                u(1,0),
                x(1,0)*sin(x(3,0)),
                u(0,0)/x(1,0) - k_s*( x(1,0)*cos(x(3,0))/(1-x(2,0)*k_s) )
                    ).finished();
            if (x.hasNaN() || dx.hasNaN()){
                std::cout << "k_s " << k_s << std::endl;
                std::cout << "x " << x << std::endl;
                std::cout << "dx " << dx << std::endl;
                throw std::runtime_error("Nan in f(x,u,i)");
            }
            return x+dx*dt;
        }
        Matrix df_dx(const Matrix x, const Matrix u, const int i){
            Scalar k_s = curvature_fun(x(0,0));
            // dfdx
            Matrix A = (Matrix(n,n) <<
                 0, cos(x(3,0))/(-k_s*x(2,0) + 1), k_s*x(1,0)*cos(x(3,0))/sqr(-k_s*x(2,0) + 1), -x(1,0)*sin(x(3,0))/(-k_s*x(2,0) + 1),
                 0, 0, 0, 0,
                 0, sin(x(3,0)), 0, x(1,0)*cos(x(3,0)),
                 0, -k_s*cos(x(3,0))/(-k_s*x(2,0) + 1) - u(0,0)/sqr(x(1,0)), -k_s*k_s*x(1,0)*cos(x(3,0))/sqr(-k_s*x(2,0) + 1), k_s*x(1,0)*sin(x(3,0))/(-k_s*x(2,0) + 1)
                                        ).finished();

            if (A.hasNaN()){
                throw std::runtime_error("Nan in df_dx(x,u,i)");
            }

            return Matrix::Identity(n,n) + A*dt;
        }
        Matrix df_du(const Matrix x, const Matrix u, const int i){

            Matrix B = (Matrix(n,m) <<
                0, 0,
                0, 1,
                0, 0,
                1/x(1,0), 0
                                        ).finished();
            if (B.hasNaN()){
                throw std::runtime_error("Nan in df_du(x,u,i)");
            }
            return B*dt;
        }

        // collision constraint function
        Scalar h(const Matrix x_i, const Matrix x_j){
            return -sqr(x_i(0,0)-x_j(0,0)) - sqr(x_i(2,0)-x_j(2,0)) + car_size*car_size;
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
            (mtx << 0,1.0+i*0.1, 0.2, 0.0 ).finished();
            return mtx;
        }
        Matrix J(const Matrix x_k, const Matrix u_k_i, int i){
            // int j = 1-i;
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
