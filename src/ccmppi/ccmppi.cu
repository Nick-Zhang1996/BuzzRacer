// cuda code for CC-MPPI with kinematic bicycle model 

#include <curand_kernel.h>
#define SAMPLE_COUNT %(SAMPLE_COUNT)s
#define HORIZON %(HORIZON)s

#define CONTROL_DIM %(CONTROL_DIM)s
#define STATE_DIM %(STATE_DIM)s
#define RACELINE_LEN %(RACELINE_LEN)s
#define CURAND_KERNEL_N %(CURAND_KERNEL_N)s
// ratio of sampled trajectory to utilize CC
#define CC_RATIO %(CC_RATIO)s
// ratio of simulations that use zero reference control
#define ZERO_REF_CTRL_RATIO %(ZERO_REF_CTRL_RATIO)s
// control cost matrix R
#define CONTROL_COST_MTX_R_1 %(R1)s
#define CONTROL_COST_MTX_R_2 %(R2)s

#define CAR_CAF %(Caf)s
#define CAR_CAR %(Car)s
#define CAR_M %(car_m)s
#define CAR_IZ %(car_Iz)s
#define CAR_LF %(car_lf)s
#define CAR_LR %(car_lr)s

#define %(MODEL_NAME)s

#define MODE_CC 1
#define MODE_NOCC 2
#define MODE_ZERO_REF 3
#define MODE_REF 4

#define PARAM_LR 0.036
#define PARAM_L 0.09

#define TEMPERATURE %(TEMPERATURE)s
#define DT %(DT)s
#define MAX_V %(MAX_V)s

#define PI 3.141592654f

#define RACELINE_DIM 6

#define RACELINE_X 0
#define RACELINE_Y 1
#define RACELINE_HEADING 2
#define RACELINE_V 3
#define RACELINE_LEFT_BOUNDARY 4
#define RACELINE_RIGHT_BOUNDARY 5

// one discretization step is around 1cm
#define RACELINE_SEARCH_RANGE 10

__device__
float evaluate_step_cost( float* state, float* u,float in_raceline[][RACELINE_DIM],int idx0, int* last_u);
__device__
float evaluate_terminal_cost( float* state,float* x0, float in_raceline[][RACELINE_DIM]);
__device__
float evaluate_collision_cost( float* state, float* opponent_pos);
__device__
void find_closest_id(float* state, float in_raceline[][RACELINE_DIM], int guess, int range, int* ret_idx, float* ret_dist);

__device__
void make_B(float vf, float* buffer);
__device__
void make_A(float vf, float* buffer);
__device__
void make_R(float angle,float* buffer);
__device__
void print_matrix(float* mtx, int m, int n, char* text);

__device__
void _evaluate_control_sequence(
    float* out_cost,
    float* out_control,
    float* x0, 
    float* in_ref_control, 
    float* limits, 
    float* in_epsilon, 
    float in_raceline[][RACELINE_DIM], 
    float opponents_prediction[][HORIZON+1][2],
    int opponent_count,
    float* Ks, float* As, float* Bs, int mode_cc, int mode_ref_control);
// cost for going off track
__device__
float evaluate_boundary_cost( float* state, float* x0, float in_raceline[][RACELINE_DIM], int* u_estimate);

// forward kinematics model by one step
#ifdef KINEMATIC_MODEL
__device__
void forward_kinematics( float* x, float* u);
#elif defined DYNAMIC_MODEL
// forward dynamic model by one step
__device__
void forward_dynamics( float* x, float* u);
#endif


// calculate matrix multiplication
// A is assumed to be a stack of 2d matrix, offset instructs the index of 2d matrix to use
// out(m*p) = A_offset(m*n) @ B(n*p), A matrix start from A+offset*n*m
__device__
void matrix_multiply_helper( float* A, int offset, float* B, int m, int n, int p, float* out){
  for (int i=0; i<m; i++){
    for (int j=0; j<p; j++){
      out[i*p + j] = 0;
      for (int k=0; k<n; k++){
        out[i*p + j] += A[offset*n*m + i*n + k] * B[k*p + j];
      }
    }
  }
}

__device__
void matrix_multiply_scalar(float* A, float coeff, int m, int n){
  for (int i=0; i<m; i++){
    for (int j=0; j<n; j++){
      A[i*n + j] *= coeff;
    }
  }

}

// element wise addition, out can be either A or B or a new buffer
__device__
void matrix_addition_helper(float* A, float* B, int m, int n, float* out){
  for (int i=0; i<m; i++){
    for (int j=0; j<n; j++){
      out[i*n + j] = A[i*n + j] + B[i*n + j];
    }
  }
}

// test matrix multiply helper
__device__
void test_matrix_multiplication(){
  float a[9] = {1,2,3,4,5,6,7,8,9};
  float b[6] = {1,4,2,5,3,6};
  float ret[6];
  matrix_multiply_helper(a,0,b,3,3,2,ret);
  float error = pow(ret[0]-14,2.0) + pow(ret[1]-32,2.0) + pow(ret[2]-32,2.0);
  printf("error = %%.2f \n",error);

}

// evaluate step cost based on target state x_goal,current state x and control u
// in_ref_control: dim horizon*control_dim
// in_epsilon: dim samples*horizon*control_dim, will be updated so that in_ref_control + in_epsilon respects limits
// in_raceline is 2d array of size (RACELINE_LEN,4), the first dimension denote different control points, the second denote data, 0:x, 1:y, 2:heading(radian), 3:ref velocity
extern "C" {
__global__
void evaluate_control_sequence(
    float* out_cost,
    float* out_control,
    float* x0, 
    float* in_ref_control, 
    float* limits, 
    float* in_epsilon, 
    float in_raceline[][RACELINE_DIM], 
    float opponents_prediction[][HORIZON+1][2],
    int opponent_count,
    float* Ks, float* As, float* Bs){

  // get global thread id
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id>=SAMPLE_COUNT){
    return;
  }

  if (id <= int(CC_RATIO * SAMPLE_COUNT)){
    if (id <= int(CC_RATIO * SAMPLE_COUNT * ZERO_REF_CTRL_RATIO)){
      _evaluate_control_sequence(out_cost, out_control, x0, in_ref_control, limits, in_epsilon, in_raceline, opponents_prediction, opponent_count, Ks, As, Bs, MODE_CC, MODE_ZERO_REF);
    } else {
      _evaluate_control_sequence(out_cost, out_control, x0, in_ref_control, limits, in_epsilon, in_raceline, opponents_prediction, opponent_count, Ks, As, Bs, MODE_CC, MODE_REF);
    }
  } else {

    if (id <= int(CC_RATIO * SAMPLE_COUNT + (1.0-CC_RATIO) * SAMPLE_COUNT * ZERO_REF_CTRL_RATIO)){
      _evaluate_control_sequence(out_cost, out_control, x0, in_ref_control, limits, in_epsilon, in_raceline, opponents_prediction, opponent_count, Ks, As, Bs, MODE_NOCC, MODE_ZERO_REF);
    } else {
      _evaluate_control_sequence(out_cost, out_control, x0, in_ref_control, limits, in_epsilon, in_raceline, opponents_prediction, opponent_count, Ks, As, Bs, MODE_NOCC, MODE_REF);
    }
  }

}
//extern C
}


// evaluate step cost based on target state x_goal,current state x and control u
// in_ref_control: dim horizon*control_dim
// in_epsilon: dim samples*horizon*control_dim, will be updated so that in_ref_control + in_epsilon respects limits
// in_raceline is 2d array of size (RACELINE_LEN,4), the first dimension denote different control points, the second denote data, 0:x, 1:y, 2:heading(radian), 3:ref velocity
__device__
void _evaluate_control_sequence(
    float* out_cost,
    float* out_control,
    float* x0, 
    float* in_ref_control, 
    float* limits, 
    float* in_epsilon, 
    float in_raceline[][RACELINE_DIM], 
    float opponents_prediction[][HORIZON+1][2],
    int opponent_count,
    float* Ks, float* As, float* Bs, int mode_cc, int mode_ref_control){

  // get global thread id
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id>=SAMPLE_COUNT){
    return;
  }

  // prepare state variables
  float x[STATE_DIM];
  float y[STATE_DIM];
  float _limits[CONTROL_DIM*2];
  // copy to local state
  // NOTE possible time saving by copy to local memory
  for (int i=0; i<STATE_DIM; i++){
    x[i] = *(x0 + i);
    y[i] = 0;
  }
  for (int i=0; i<CONTROL_DIM*2; i++){
    _limits[i] = limits[i];
  }


  float cost = 0;
  // used as estimate to find closest index on raceline
  int last_u = -1;

  // record index of initial state
  int idx0;
  float unused;
  find_closest_id(x0,in_raceline,-1,0,&idx0,&unused);

  // run simulation
  // loop over time horizon
  float total_collision_cost = 0.0;
  for (int i=0; i<HORIZON; i++){
    float _u[CONTROL_DIM];
    float* u = _u;

    // calculate control
    // u = v(ref control) + K * y(cc feedback) + epsilon (noise)
    // feedback
    if (mode_cc == MODE_CC){
      matrix_multiply_helper(Ks, i, y, CONTROL_DIM, STATE_DIM, 1, u);
    }

    for (int j=0; j<CONTROL_DIM; j++){
      // calculate control
      float val = in_epsilon[id*HORIZON*CONTROL_DIM + i*CONTROL_DIM + j];
      if (mode_ref_control == MODE_REF){
        val += in_ref_control[i*CONTROL_DIM + j];
      }
      if (mode_cc == MODE_CC){
        val += u[j];
      }

      // apply constrain on control input
      val = val < _limits[j*CONTROL_DIM]? _limits[j*CONTROL_DIM]:val;
      val = val > _limits[j*CONTROL_DIM+1]? _limits[j*CONTROL_DIM+1]:val;

      // set control
      u[j] = val;
      out_control[id*HORIZON*CONTROL_DIM + i*CONTROL_DIM + j] = u[j];

    }

    /*
    if (id == 0){
      printf("states = %%7.4f, %%7.4f, %%7.4f, %%7.4f, ctrl =  %%7.4f, %%7.4f \n", x[0], x[1], x[2], x[3], u[0], u[1]);
    }
    */
    // step forward dynamics, update state x in place
#ifdef KINEMATIC_MODEL
    forward_kinematics(x, u);
#elif defined DYNAMIC_MODEL
    forward_dynamics(x, u);
#endif

    // evaluate step cost (crosstrack error and velocity deviation)
    // corresponds to q(x)
    float step_cost = evaluate_step_cost(x,u,in_raceline,idx0,&last_u);
    // cost related to obstacle collision avoidance / opponent avoidance
    // note this does not predict opponent
    // i: prediction step
    if (i < HORIZON){
      for (int j=0; j<opponent_count; j++){
        float this_collision_cost = evaluate_collision_cost(x,opponents_prediction[j][i]);
        // * horizon to normalize
        cost += this_collision_cost*HORIZON;
        total_collision_cost += this_collision_cost;
      }
    }

    int temp_index;
    float temp_dist;
    find_closest_id(x,in_raceline,last_u,RACELINE_SEARCH_RANGE, &temp_index, &temp_dist);

    cost += step_cost;

    // evaluate track boundary cost
    int u_estimate = -1;
    cost += evaluate_boundary_cost(x,x0,in_raceline, &u_estimate);
    // add control cost
    float* eps = in_epsilon + id*HORIZON*CONTROL_DIM + i*CONTROL_DIM;
    // 1/2*eps' * R * eps
    cost += 0.5 * (eps[0]*eps[0]*CONTROL_COST_MTX_R_1 + eps[1]*eps[1]*CONTROL_COST_MTX_R_2);
    // v * R * eps
    cost += in_ref_control[i*CONTROL_DIM + 0] * CONTROL_COST_MTX_R_1 * eps[0];
    cost += in_ref_control[i*CONTROL_DIM + 1] * CONTROL_COST_MTX_R_2 * eps[2];
    // 1/2 v' * R * v
    cost += in_ref_control[i*CONTROL_DIM + 0] * in_ref_control[i*CONTROL_DIM + 0] * CONTROL_COST_MTX_R_1;
    cost += in_ref_control[i*CONTROL_DIM + 1] * in_ref_control[i*CONTROL_DIM + 1] * CONTROL_COST_MTX_R_2;


    // update y
    // y = A * y + B * epsilon
    if (mode_cc == MODE_CC){
      float temp[STATE_DIM];
      float temp2[STATE_DIM];
      float* this_epsilon = in_epsilon + id*HORIZON*CONTROL_DIM + i*CONTROL_DIM;

      matrix_multiply_helper(As, i, y, STATE_DIM, STATE_DIM, 1, temp);
      matrix_multiply_helper(Bs, i, this_epsilon, STATE_DIM, CONTROL_DIM, 1, temp2);
      for (int k=0; k<STATE_DIM; k++){
        y[k] = temp[k] + temp2[k];
      }
    }

    u += CONTROL_DIM;

  }


  float terminal_cost = evaluate_terminal_cost(x,x0,in_raceline);
  cost += terminal_cost;
  out_cost[id] = cost;

}

// find closest id in the index range (guess - range, guess + range)
// if guess is -1 then the entire spectrum will be searched
__device__
void find_closest_id(float* state, float in_raceline[][RACELINE_DIM], int guess, int range, int* ret_idx, float* ret_dist){
  float x = state[0];
  float y = state[1];
  float val;

  int idx = 0;
  float current_min = 1e6;

  int start, end;
  if (guess == -1){
    start = 0;
    end = RACELINE_LEN;
  } else {
    start = guess - range;
    end = guess + range;
  }

  for (int k=start;k<end;k++){
    int i = (k + RACELINE_LEN) %% RACELINE_LEN;
    val = (x-in_raceline[i][0])*(x-in_raceline[i][0]) + (y-in_raceline[i][1])*(y-in_raceline[i][1]);
    if (val < current_min){
      idx = i;
      current_min = val;
    }
  }

  *ret_idx = idx;
  *ret_dist = sqrtf(current_min);
  return;

}


__device__
float evaluate_step_cost( float* state, float* u, float in_raceline[][RACELINE_DIM], int idx0, int* last_u){
  int idx;
  float dist, cost;
  cost = 0;
  // velocity cost
  //dv = state[2] - in_raceline[idx][3];
  // additional penalty on negative velocity 
  #ifdef KINEMATIC_MODEL
  if (state[2] < 0){
    cost += 0.1;
  }
  #elif defined DYNAMIC_MODEL
  if (state[3] < 0){
    cost += 0.1;
  }
  #endif

  // progress cost
  find_closest_id(state,in_raceline,idx0+80,80,&idx,&dist);
  // update estimate of closest index on raceline
  *last_u = idx;

  // wrapping
  // *0.01: convert index difference into length difference
  // length of raceline is roughly 10m, with 1000 points roughly 1d_index=0.01m
  cost =  (1.0-1.0*float((idx - idx0 + RACELINE_LEN) %% RACELINE_LEN)*0.01)*3.3;
  cost += dist*dist*10;

  //return cost;
  return 0.0;
}

__device__
float evaluate_terminal_cost( float* state,float* x0, float in_raceline[][RACELINE_DIM]){
  int idx0,idx;
  float dist,cost;

  // we don't need distance info for initial state, 
  //dist is put in as a dummy variable, it is immediately overritten
  find_closest_id(x0,in_raceline,-1,0,&idx0,&dist);
  find_closest_id(state,in_raceline,idx0+80,80,&idx,&dist);

  // wrapping
  // *0.01: convert index difference into length difference
  // length of raceline is roughly 10m, with 1000 points roughly 1d_index=0.01m
  cost =  (1.0-1.0*float((idx - idx0 + RACELINE_LEN) %% RACELINE_LEN)*0.01)*3.3;
  cost += dist*dist*500;
  return cost;
  // NOTE ignoring terminal cost
  //return 0.0;
}

// NOTE potential improvement by reusing idx result from other functions
// u_estimate is the estimate of index on raceline that's closest to state
__device__
float evaluate_boundary_cost( float* state, float* x0, float in_raceline[][RACELINE_DIM], int* u_estimate){
  int idx;
  float dist;

  // performance barrier FIXME
  find_closest_id(state,in_raceline,*u_estimate, RACELINE_SEARCH_RANGE, &idx,&dist);
  *u_estimate = idx;
  
  float tangent_angle = in_raceline[idx][4];
  float raceline_to_point_angle = atan2f(in_raceline[idx][1] - state[1], in_raceline[idx][0] - state[0]) ;
  float angle_diff = fmodf(raceline_to_point_angle - tangent_angle + PI, 2*PI) - PI;

  float cost;

  if (angle_diff > 0.0){
    // point is to left of raceline
    //cost = (dist +0.05> in_raceline[idx][4])? 0.3:0.0;
    cost = (dist +0.05> in_raceline[idx][4])? 2000.0:0.0;
  } else {
    //cost = (dist +0.05> in_raceline[idx][5])? 0.3:0.0;
    cost = (dist +0.05> in_raceline[idx][5])? 2000.0:0.0;
  }

  return cost;
}


// forward dynamics using kinematic model
// state: X,Y,v_forward,psi
// control : throttle, steering (left +)
__device__
void forward_kinematics(float* state, float* u){
  float throttle,steering;
  float velocity, psi;

  velocity = state[2];
  psi = state[3];

  throttle = u[0];
  steering = u[1];

  float beta = atanf(tanf(steering)*PARAM_LR / PARAM_L);
  float dx = velocity * cosf(psi + beta) * DT;
  float dy = velocity * sinf(psi + beta) * DT;
  float dvelocity;
  if (velocity > MAX_V){
    dvelocity = -0.01;
  } else {
    dvelocity = throttle * DT;

  }

  float dpsi = velocity / PARAM_LR * sinf(beta) * DT;

  state[0] += dx;
  state[1] += dy;
  state[2] += dvelocity;
  state[3] += dpsi;

}

__device__
void forward_dynamics(float* state, float* u){
  // get global thread id
  //int id = blockIdx.x * blockDim.x + threadIdx.x;
  // replicate DynamicSimulator.advanceDYnamics
  float x = state[0];
  float y = state[1];
  float psi = state[2];
  float vf = state[3];
  float vs = state[4];
  float omega = state[5];
  float vx = vf*cosf(psi)-vs*sinf(psi);
  float vy = vf*sinf(psi)+vs*cosf(psi);
  //float throttle = u[0];
  //float steering = u[1];
  float A[6*6],B[6*2],R[6*6],buffer[6*6];
  float state_buffer[6],state_buffer2[6];
  float sim_states[6];
  float kinematic_state[4];

  // fallback to kinematic model if velocity is too low to avoid singularity
  /*
  if (vf < 0.4){
    kinematic_state[0] = x;
    kinematic_state[1] = y;
    kinematic_state[2] = vf;
    kinematic_state[3] = psi;
    forward_kinematics(kinematic_state, u);
    state[0] = kinematic_state[0];
    state[1] = kinematic_state[1];
    state[2] = kinematic_state[3];
    state[3] = kinematic_state[2];
    state[4] = 0;
    state[5] = 0;
    return;
  }
  */



  sim_states[0] = x;
  sim_states[1] = vx;
  sim_states[2] = y;
  sim_states[3] = vy;
  sim_states[4] = psi;
  sim_states[5] = omega;


  make_A(vf,A);// 6*6
  make_B(vf,B);// 6*2
  make_R(-psi,R);//6*6
  /*
  if (id==0){
    printf("pre update: \n");
    printf("car_state = np.array([%%.4f, %%.4f, %%.4f, %%.4f, %%.4f, %%.4f])\n", x,y,psi,vf,vs,omega);
    printf("sim_state = np.array([%%.4f, %%.4f, %%.4f, %%.4f, %%.4f, %%.4f])\n", x,vx,y,vy,psi,omega);
    printf("control = np.array([%%.4f, %%.4f])\n", u[0], u[1]);
    print_matrix(A,6,6,"A");
    print_matrix(B,6,2,"B");
    print_matrix(R,6,6,"R(-psi)");
  }
  */
  matrix_multiply_helper( A, 0, R, 6, 6, 6, buffer);
  /*
  if (id==0){
    print_matrix(buffer, 6, 6," A@R");
  }
  */
  matrix_multiply_helper( buffer, 0, sim_states, 6, 6, 1, state_buffer);
  /*
  if (id==0){
    print_matrix(state_buffer, 6, 1," A@R@sim_states");
  }
  */
  matrix_multiply_helper( B, 0, u, 6, 2, 1, state_buffer2);
  /*
  if (id==0){
    print_matrix(state_buffer2, 6, 1," B@u");
  }
  */
  matrix_addition_helper( state_buffer, state_buffer2, 6,1, state_buffer);
  /*
  if (id==0){
    print_matrix(state_buffer, 6, 1,"A@R@sim_states+ B@u");
  }
  */
  make_R(psi,R);//6*6
  /*
  if (id==0){
    print_matrix(R,6,6,"R(psi)");
  }
  */
  matrix_multiply_helper(R, 0, state_buffer, 6, 6, 1, state_buffer2);
  /*
  if (id==0){
    print_matrix(state_buffer, 6, 1,"R(A@R@sim_states+ B@u)");
  }
  */
  matrix_multiply_scalar(state_buffer, DT, 6, 1);
  /*
  if (id==0){
    print_matrix(state_buffer, 6, 1,"R(A@R@sim_states+ B@u)*dt");
  }
  */
  matrix_addition_helper(state_buffer, sim_states, 6, 1, sim_states);
  //x
  state[0] = sim_states[0];
  //y
  state[1] = sim_states[2];
  //psi
  state[2] = sim_states[4];
  //vf
  state[3] = sim_states[1]*cosf(sim_states[4]) + sim_states[3]*sinf(sim_states[4]);
  //vs
  state[4] = -sim_states[1]*sinf(sim_states[4]) + sim_states[3]*cosf(sim_states[4]);
  //omage
  state[5] = sim_states[5];
  /*
  if (id==0){
    printf("post update: \n");
    printf("car_state = np.array([%%.4f, %%.4f, %%.4f, %%.4f, %%.4f, %%.4f])\n", state[0], state[1], state[2], state[3], state[4], state[5]);
    printf("sim_states = np.array([%%.4f, %%.4f, %%.4f, %%.4f, %%.4f, %%.4f])\n", sim_states[0], sim_states[1], sim_states[2], sim_states[3], sim_states[4], sim_states[5]);
  }
  */
}

//buffer: 6*2=12
__device__
void make_B(float vf, float* buffer){
  *(buffer+0) = 0;
  *(buffer+1) = 0;

  *(buffer+2*1+0) = 1;
  *(buffer+2*1+1) = 0;

  *(buffer+2*2+0) = 0;
  *(buffer+2*2+1) = 0;

  *(buffer+2*3+0) = 0;
  *(buffer+2*3+1) = 2*CAR_CAF/CAR_M;

  *(buffer+2*4+0) = 0;
  *(buffer+2*4+1) = 0;

  *(buffer+2*5+0) = 0;
  *(buffer+2*5+1) = 2*CAR_LF*CAR_CAF/CAR_IZ;
  // get global thread id
  /*
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id==0){
    printf("debug= %%.4f\n",*(buffer+2));
    print_matrix(buffer,6,2,"B(make_B)");
  }
  */

}

//buffer: 6*6=36
__device__
void make_A(float vf, float* buffer){
  *(buffer+0) = 0;
  *(buffer+1) = 1;
  *(buffer+2) = 0;
  *(buffer+3) = 0;
  *(buffer+4) = 0;
  *(buffer+5) = 0;

  *(buffer+6*1+0) = 0;
  *(buffer+6*1+1) = 0;
  *(buffer+6*1+2) = 0;
  *(buffer+6*1+3) = 0;
  *(buffer+6*1+4) = 0;
  *(buffer+6*1+5) = 0;

  *(buffer+6*2+0) = 0;
  *(buffer+6*2+1) = 0;
  *(buffer+6*2+2) = 0;
  *(buffer+6*2+3) = 1;
  *(buffer+6*2+4) = 0;
  *(buffer+6*2+5) = 0;
  
  *(buffer+6*3+0) = 0;
  *(buffer+6*3+1) = 0;
  *(buffer+6*3+2) = 0;
  *(buffer+6*3+3) = -(2*CAR_CAF + 2*CAR_CAR)/(CAR_M*vf);
  *(buffer+6*3+4) = 0;
  *(buffer+6*3+5) = -vf-(2*CAR_CAF*CAR_LF-2*CAR_CAR*CAR_LR)/(CAR_M*vf);

  *(buffer+6*4+0) = 0;
  *(buffer+6*4+1) = 0;
  *(buffer+6*4+2) = 0;
  *(buffer+6*4+3) = 0;
  *(buffer+6*4+4) = 0;
  *(buffer+6*4+5) = 1;
  
  *(buffer+6*5+0) = 0;
  *(buffer+6*5+1) = 0;
  *(buffer+6*5+2) = 0;
  *(buffer+6*5+3) = -(2*CAR_LF*CAR_CAF-2*CAR_LR*CAR_CAR)/(CAR_IZ*vf);
  *(buffer+6*5+4) = 0;
  *(buffer+6*5+5) = -(2*CAR_LF*CAR_LF*CAR_CAF+2*CAR_LR*CAR_LR*CAR_CAR)/(CAR_IZ*vf);

}

// make active rotation matrix
// buffer: 6*6=36
__device__
void make_R(float angle,float* buffer){
  float c = cosf(angle);
  float s = sinf(angle);
  *(buffer+0) = c;
  *(buffer+1) = 0;
  *(buffer+2) = -s;
  *(buffer+3) = 0;
  *(buffer+4) = 0;
  *(buffer+5) = 0;

  *(buffer+6*1+0) = 0;
  *(buffer+6*1+1) = c;
  *(buffer+6*1+2) = 0;
  *(buffer+6*1+3) = -s;
  *(buffer+6*1+4) = 0;
  *(buffer+6*1+5) = 0;

  *(buffer+6*2+0) = s;
  *(buffer+6*2+1) = 0;
  *(buffer+6*2+2) = c;
  *(buffer+6*2+3) = 0;
  *(buffer+6*2+4) = 0;
  *(buffer+6*2+5) = 0;

  *(buffer+6*3+0) = 0;
  *(buffer+6*3+1) = s;
  *(buffer+6*3+2) = 0;
  *(buffer+6*3+3) = c;
  *(buffer+6*3+4) = 0;
  *(buffer+6*3+5) = 0;

  *(buffer+6*4+0) = 0;
  *(buffer+6*4+1) = 0;
  *(buffer+6*4+2) = 0;
  *(buffer+6*4+3) = 0;
  *(buffer+6*4+4) = 1;
  *(buffer+6*4+5) = 0;

  *(buffer+6*5+0) = 0;
  *(buffer+6*5+1) = 0;
  *(buffer+6*5+2) = 0;
  *(buffer+6*5+3) = 0;
  *(buffer+6*5+4) = 0;
  *(buffer+6*5+5) = 1;
}

__device__
void print_matrix(float* mtx, int m, int n, char* text){
  printf(text);
  printf(" = \n");
  for (int i=0; i<m; i++){
    for (int j=0; j<n; j++){
      printf("%%7.3f, ",mtx[i*n+j]);
    }
    printf("\n");
  }

}

__device__
float evaluate_collision_cost( float* state, float* opponent_pos){
  //float heading = state[4];

  float dx = state[0]-opponent_pos[0];
  float dy = state[1]-opponent_pos[1];
  //float cost = 5.0*(0.1 - sqrtf(dx*dx + dy*dy)) ;
  float cost = 5.0*(0.1 - sqrtf(dx*dx + dy*dy)) ;
  cost = cost>0? cost:0;

  return cost ;
}


// curand funtions
__device__ curandState_t* curand_states[CURAND_KERNEL_N];

extern "C" {
__global__ void init_curand_kernel(int seed){
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= SAMPLE_COUNT*HORIZON*CONTROL_DIM) return;

  curandState_t* s = new curandState_t;
  if (s != 0) {
    curand_init(seed, id, 0, s);
  } else {
    printf("error initializing curand kernel\n");
  }

  curand_states[id] = s;
}

// limits: [u0_low,u0_high,u1_low,u1_high...]
__global__ void generate_random_normal(float *values,float* scales){
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  
  // failsafe, should never be true
  if (id >= CURAND_KERNEL_N) {return;}

  float _scales[CONTROL_DIM*2];
  for (int i=0; i<CONTROL_DIM*2; i++){
    _scales[i] = scales[i];
  }

  curandState_t s = *curand_states[id];
  int start = id*(SAMPLE_COUNT*HORIZON*CONTROL_DIM)/CURAND_KERNEL_N;
  int end = min(SAMPLE_COUNT*HORIZON*CONTROL_DIM,(id+1)*(SAMPLE_COUNT*HORIZON*CONTROL_DIM)/CURAND_KERNEL_N);
  //printf("id %%d, %%d - %%d\n",id, start, end);

  for(int i=start; i < end; i+=CONTROL_DIM ) {
    for (int j=0; j<CONTROL_DIM; j++){
      float val = curand_normal(&s) * _scales[j];
      values[i+j] = val;
    }
    
  }
  *curand_states[id] = s;

}

// extern "C"
}

