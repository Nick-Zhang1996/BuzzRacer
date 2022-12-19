// cuda code for MPPI with kinetic bicycle model

#include <curand_kernel.h>
#define SAMPLE_COUNT %(SAMPLE_COUNT)s
#define HORIZON %(HORIZON)s

#define CONTROL_DIM %(CONTROL_DIM)s
#define STATE_DIM %(STATE_DIM)s
#define RACELINE_LEN %(RACELINE_LEN)s
#define CURAND_KERNEL_N %(CURAND_KERNEL_N)s

#define OBSTACLE_RADIUS 0.08

#define PARAM_LF 0.04824
#define PARAM_LR (0.09-0.04824)
#define PARAM_L 0.09
#define PARAM_H 0.01            // center of mass height

#define PARAM_IZ 417757e-9
#define PARAM_MASS 0.1667

// old ax0 model
//#define MOTOR_PARAM_A 6.17
//#define MOTOR_PARAM_B 15.2
//#define MOTOR_PARAM_C 0.2
// new ax0 model
# define MOTOR_PARAM_A 27.42298

#define PARAM_B 2.3
#define PARAM_C 1.6
#define PARAM_D 1.1

#define PARAM_K_US 0.028521         // understeer gradient
//#define PARAM_TAU_A 0.04            // accel time constant, with old ax0 model
#define PARAM_TAU_A 0.379826        // accel time constant
#define PARAM_TAU_DELTA 0.0613      // steering time constant
#define PARAM_TAU_OMEGA 0.125744    // angular velocity time constant
//#define PARAM_K_D 0.0f              // drag coefficient, with old ax0 model
//#define PARAM_C_R 0.0f              // rotational friction coefficient, with old ax0 model
#define PARAM_K_D 0.0942299         // drag coefficient
#define PARAM_C_R 4.49905           // rotational friction coefficient


#define TEMPERATURE %(TEMPERATURE)s
#define DT %(DT)s

#define PI 3.141592654f

#define RACELINE_DIM 8

#define RACELINE_X 0
#define RACELINE_Y 1
#define RACELINE_HEADING 2
#define RACELINE_V 3
#define RACELINE_LEFT_BOUNDARY 4
#define RACELINE_RIGHT_BOUNDARY 5
#define RACELINE_SS 6
#define RACELINE_K 7

#define STATE_AX 0
#define STATE_DELTA 1
#define STATE_VX 2
#define STATE_OMEGA 3
#define STATE_ZETA 4
#define STATE_N 5
#define STATE_XI 6
#define STATE_CURVATURE 7

#define CONTROL_THROTTLE 0
#define CONTROL_STEERING 1

// one discretization step is around 1cm
#define RACELINE_SEARCH_RANGE 10


// vars
__device__ curandState_t* curand_states[CURAND_KERNEL_N];
__device__ float control_limit[2*CONTROL_DIM];
__device__ float noise_std[CONTROL_DIM];
__device__ float noise_mean[CONTROL_DIM];
__device__ float sampled_noise[SAMPLE_COUNT*HORIZON*CONTROL_DIM];
__device__ float raceline[RACELINE_LEN][RACELINE_DIM];

// device functions
__device__
float evaluate_terminal_cost( float* current_state,float* initial_state);
__device__
void find_closest_id(float* state, int guess, int* ret_idx, float* ret_k);
__device__
float evaluate_boundary_cost( float* state, int* u_estimate);
__device__
float evaluate_step_cost( float* state, float* last_u, float* u,int* last_index);
__device__
float evaluate_collision_cost( float* state, float* opponent_traj);
__device__
void forward_dynamics( float* state, float* u, float curvature);
__device__
float tire_curve( float slip);

extern "C" {
__global__ void init_curand_kernel(int seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= SAMPLE_COUNT * HORIZON * CONTROL_DIM) return;

    curandState_t *s = new curandState_t;
    if (s != 0) {
        curand_init(seed, id, 0, s);
    } else {
        printf("error initializing curand kernel\n");
    }

    curand_states[id] = s;
}
__global__ void set_control_limit(float *in_control_limit) {
    for (int i = 0; i < sizeof(control_limit); i++) { control_limit[i] = in_control_limit[i]; }
}
__global__ void set_noise_cov(float *in_noise_cov) {
    for (int i = 0; i < sizeof(noise_std); i++) { noise_std[i] = sqrtf(in_noise_cov[i]); }
    //printf("std: %%.2f, %%.2f \n",noise_std[0],noise_std[1]);
}
__global__ void set_noise_mean(float *in_noise_mean) {
    for (int i = 0; i < sizeof(noise_mean); i++) { noise_mean[i] = sqrtf(in_noise_mean[i]); }
}

__global__ void set_raceline(float *in_raceline) {
    for (int i = 0; i <RACELINE_LEN;
    i++){
        raceline[i][0] = in_raceline[i * RACELINE_DIM + 0];
        raceline[i][1] = in_raceline[i * RACELINE_DIM + 1];
        raceline[i][2] = in_raceline[i * RACELINE_DIM + 2];
        raceline[i][3] = in_raceline[i * RACELINE_DIM + 3];
        raceline[i][4] = in_raceline[i * RACELINE_DIM + 4];
        raceline[i][5] = in_raceline[i * RACELINE_DIM + 5];
        raceline[i][6] = in_raceline[i * RACELINE_DIM + 6];
        raceline[i][7] = in_raceline[i * RACELINE_DIM + 7];
    }
}

__global__ void generate_control_noise() {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    // failsafe, should never be true
    if (id >= CURAND_KERNEL_N) { return; }

    float _scales[CONTROL_DIM * 2];
    for (int i = 0; i < sizeof(_scales); i++) { _scales[i] = noise_std[i]; }

    curandState_t s = *curand_states[id];
    int start = id *(SAMPLE_COUNT * HORIZON * CONTROL_DIM)/CURAND_KERNEL_N;
    int end = min( SAMPLE_COUNT * HORIZON * CONTROL_DIM, (id
    +1)*(SAMPLE_COUNT * HORIZON * CONTROL_DIM)/CURAND_KERNEL_N);
    //printf("id %%d, %%d - %%d\n",id, start, end);

    for (int i = start; i < end; i +=CONTROL_DIM ) {
        for (int j = 0; j <CONTROL_DIM;
        j++){
            float val = curand_normal(&s) * _scales[j] + noise_mean[j];
            sampled_noise[i + j] = val;
            // DEBUG
            //out_values[i+j] = val;
        }
    }
    *curand_states[id] = s;

}
// evaluate sampled control sequences
// x0: x,y,heading, v_forward, v_sideways, omega
// u0: current control to penalize control time rate
// ref_control: samples*horizon*control_dim
// out_cost: samples 
// out_trajectories: output trajectories, samples*horizon*n
// opponent_count: integer
// opponent_traj: opponent_count * prediction_horizon * 2(x,y)
//__global__ void evaluate_control_sequence(float* in_x0, float* in_u0, float* ref_dudt, float* out_cost, float* out_dudt, float* out_trajectories){
__global__ void evaluate_control_sequence(float *in_x0, float *in_u0, float *ref_dudt, float *out_cost, float *out_dudt,
                                          int opponent_count, float *in_opponent_traj) {
    // get global thread id
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >=SAMPLE_COUNT){
        return;
    }

    float x[STATE_DIM];
    // copy to local state
    // NOTE possible time saving by copy to local memory
    for (int i = 0; i <STATE_DIM;
    i++){
        x[i] = *(in_x0 + i);
    }

    // initialize cost
    float cost = 0;
    // used as estimate to find closest index on raceline
    int last_index = -1;
    float last_u[CONTROL_DIM];
    for (int i = 0; i <CONTROL_DIM;
    i++){
        last_u[i] = *(in_u0 + i);
    }

    /*
    if (id == 0){
      printf("last u0=%%.2f",last_u[1]*180.0/PI);
    }
    */

    // run simulation
    // loop over time horizon
    for (int i = 0; i <HORIZON;
    i++){
        float _u[CONTROL_DIM];
        float *u = _u;

        // apply constrain on control input
        for (int j = 0; j <CONTROL_DIM;
        j++){
            // NOTE control is variation
            float dudt = (ref_dudt[i * CONTROL_DIM + j] +
                          sampled_noise[id * HORIZON * CONTROL_DIM + i * CONTROL_DIM + j]);
            float val = last_u[j] + dudt * DT;
            val = val < control_limit[j * CONTROL_DIM] ? control_limit[j * CONTROL_DIM] : val;
            val = val > control_limit[j * CONTROL_DIM + 1] ? control_limit[j * CONTROL_DIM + 1] : val;
            //out_dudt[id*HORIZON*CONTROL_DIM + i*CONTROL_DIM + j] = val;
            out_dudt[id * HORIZON * CONTROL_DIM + i * CONTROL_DIM + j] = (val - last_u[j]) / DT;
            u[j] = val;
        }
        /*
        if(id==0 && i==0){
          printf("u-1 = %%.2f\n",last_u[1]*180.0/PI);
        }
        */

        // step forward dynamics, update state x in place
        int idx;
        float k;
        find_closest_id(x, last_index, &idx, &k);
        forward_dynamics(x, u, k);
        /*
        for (int j=0; j<STATE_DIM; j++){
          out_trajectories[id*HORIZON*STATE_DIM + i*STATE_DIM + j] = x[j];
        }
        */

        // evaluate step cost
        if (i == 0) {
            cost += evaluate_step_cost(x, last_u, u, &last_index);
        } else {
            cost += evaluate_step_cost(x, u, u, &last_index);
        }
        cost += evaluate_boundary_cost(x, &last_index);
        for (int k = 0; k <CONTROL_DIM;
        k++){
            last_u[k] = u[k];
        }

        u += CONTROL_DIM;

    }

    float terminal_cost = evaluate_terminal_cost(x, in_x0);
    cost += evaluate_terminal_cost(x, in_x0);
    cost += terminal_cost;
    out_cost[id] = cost;
}

//extern c
}

// todo: VERIFY THAT GLOBAL X AND Y NOT NEEDED
// x: ax, delta, vx, Omega, zeta, n, xi
// u: throttle, steering
__device__
void forward_dynamics(float *state, float *u, float curvature) {
    float ax, delta, vx, Omega, zeta, n, xi;
//    float curvature;
    float d_ax, d_delta, d_vx, d_Omega, d_zeta, d_n, d_xi;
    float ax0, throttle, delta0;
    ax = state[STATE_AX];
    delta = state[STATE_DELTA];
    vx = state[STATE_VX];
    Omega = state[STATE_OMEGA];
    zeta = state[STATE_ZETA];
    n = state[STATE_N];
    xi = state[STATE_XI];
//    curvature = state[STATE_CURVATURE];

    throttle = u[CONTROL_THROTTLE];
    // old ax0 model from motor model
//    ax0 = MOTOR_PARAM_A * (throttle - vx / MOTOR_PARAM_B - MOTOR_PARAM_C);
    // new throttle -> ax0 model
    ax0 = MOTOR_PARAM_A * throttle;
    delta0 = u[CONTROL_STEERING];

    d_Omega = 1 / PARAM_TAU_OMEGA * (vx / PARAM_L * (delta - PARAM_K_US) - Omega);
    d_vx = ax - PARAM_K_D / PARAM_MASS * vx * vx - PARAM_C_R * vx;
    d_ax = 1 / PARAM_TAU_A * (ax0 - ax);
    d_delta = 1 / PARAM_TAU_DELTA * (delta0 - delta);
    d_zeta = -(vx * cosf(xi)) / (n * curvature - 1);
    d_n = vx * sinf(xi);
    d_xi = Omega + (vx * cosf(xi) * curvature) / (n * curvature - 1);

    Omega += d_Omega * DT;
    vx += d_vx * DT;
    ax += d_ax * DT;
    delta += d_delta * DT;
    zeta += d_zeta * DT;
    n += d_n * DT;
    xi += d_xi * DT;

    state[STATE_AX] = ax;
    state[STATE_DELTA] = delta;
    state[STATE_VX] = vx;
    state[STATE_OMEGA] = Omega;
    state[STATE_ZETA] = zeta;
    state[STATE_N] = n;
    state[STATE_XI] = xi;
    return;
}

__device__
float evaluate_step_cost( float* state, float* last_u, float* u,int* last_index){
    //float heading = state[4];
    int idx;
    float dist = state[STATE_N];
    float ret_k;

    find_closest_id(state,*last_index, &idx, &ret_k);



    // update estimate of closest index on raceline
//    if (threadIdx.x + blockDim.x * blockIdx.x == 100) {
//        printf("%%d\n", idx);
//    }
    *last_index = idx;


    // velocity cost
    // current FORWARD velocity - target velocity at closest ref point

    // forward vel
    float vx = state[STATE_VX];

    // velocity deviation from reference velocity profile
    float dv = vx - raceline[idx][3];
    // control change from last step, penalize to smooth control

    //float cost = dist + 1.0*dv*dv + 1.0*du_sqr;
    float cost = 3*dist*dist + 0.6*dv*dv ;
//    printf("%%f\n", dist);
    // heading cost

    // todo determine if this actually works
    float temp = fmodf(state[STATE_XI] + 3*PI, 2*PI) - PI;
    //  float temp = fmodf(raceline[idx][2] - state[STATE_HEADING] + 3*PI,2*PI) - PI;
    cost += temp*temp*2.5;
    //float cost = dist;
    // additional penalty on negative velocity
    if (vx < 0.05){
    cost += 0.2;
    }
    return cost;
}

// NOTE potential improvement by reusing idx result from other functions
// u_estimate is the estimate of index on raceline that's closest to state
__device__
float evaluate_boundary_cost( float* state,  int* u_estimate){
    int idx;
    float dist = state[STATE_N];
    float ret_k;

    // todo: find closest id needs x and y, but we're already in curvilinear so just go with existing coords?
    // not sure how zeta maps to raceline distance
    // performance barrier FIXME
    find_closest_id(state,*u_estimate,  &idx, &ret_k);

    *u_estimate = idx;

//    float tangent_angle = raceline[idx][4];
//    float raceline_to_point_angle = atan2f(raceline[idx][1] - state[STATE_Y], raceline[idx][0] - state[STATE_X]) ;
//    float angle_diff = fmodf(raceline_to_point_angle - tangent_angle + PI, 2*PI) - PI;
    float angle_diff = fmodf(state[STATE_XI] + 3*PI, 2*PI) - PI;

    float cost;

    if (angle_diff > 0.0){
    // point is to left of raceline
    cost = (dist +0.05> raceline[idx][4])? 0.3:0.0;
    } else {
    cost = (dist +0.05> raceline[idx][5])? 0.3:0.0;
    }

    return cost;
}

// find closest id in the index range (guess - range, guess + range)
// if guess is -1 then the entire spectrum will be searched
__device__
void find_closest_id(float* state, int guess, int* ret_idx, float* ret_k) {
    float zeta = state[STATE_ZETA];
    int flag = true;
    int bottom;
    int bottomInd;
    int top;
    int topInd;
    int i = guess;
//    int i = RACELINE_LEN / 2; // I hope RACELINE_LEN is an integer
    if (guess == -1) {
        bottom = 0;
        i = RACELINE_LEN / 2;
        top = RACELINE_LEN - 1;
    } else {
        bottom = i - 10;
        top = i + 10;
    }
    while (flag) {
        if (top - bottom == 1) {
            if (bottom < 0) {
                bottomInd = RACELINE_LEN + bottom;
            } else {
                bottomInd = bottom;
            }
            if (top > RACELINE_LEN) {
                topInd = top - RACELINE_LEN;
            } else {
                topInd = top;
            }
            float bottomDiff = zeta - raceline[bottomInd][RACELINE_SS];
            float topDiff = raceline[topInd][RACELINE_SS] - zeta;
            if (bottomDiff < topDiff) {
                *ret_idx = bottomInd;
                *ret_k = raceline[bottomInd][RACELINE_K];
                flag = false;
            } else {
                *ret_idx = topInd;
                *ret_k = raceline[topInd][RACELINE_K];
                flag = false;
            }
        } else {
            //        if (raceline[i][RACELINE_SS] == zeta) {
            //            *ret_idx = i;
            //            flag = false;
            if (raceline[i][RACELINE_SS] <= zeta) {
                bottom = i;
                i = (top - bottom) / 2 + bottom;
            } else if (raceline[i][RACELINE_SS] > zeta) {
                top = i;
                i = (top - bottom) / 2 + bottom;
            }
        }
    }
    return;
}


__device__
float evaluate_terminal_cost( float* current_state,float* initial_state){
  //int idx0,idx;
  //float dist;

  // we don't need distance info for initial state, 
  //dist is put in as a dummy variable, it is immediately overritten
  //find_closest_id(x0,raceline,-1,0,&idx0,&dist);
  //find_closest_id(state,raceline,-1,0,&idx,&dist);

  // wrapping
  // *0.01: convert index difference into length difference
  // length of raceline is roughly 10m, with 1000 points roughly 1d_index=0.01m
  //return -1.0*float((idx - idx0 + RACELINE_LEN) %% RACELINE_LEN)*0.01;
  // NOTE ignoring terminal cost
  return 0.0;
}

__device__
float tire_curve( float slip){
  return PARAM_D * sinf( PARAM_C * atanf( PARAM_B * slip) );

}

__device__
float evaluate_collision_cost( float* state, float* opponent_pos){
  //float heading = state[4];

  // fixme!! (don't have state_x and state_y)
//  float dx = state[STATE_X]-opponent_pos[0];
//  float dy = state[STATE_Y]-opponent_pos[1];
//
//  float cost = 5.0*(OBSTACLE_RADIUS - sqrtf(dx*dx + dy*dy)) ;

  return 0.0;
}

