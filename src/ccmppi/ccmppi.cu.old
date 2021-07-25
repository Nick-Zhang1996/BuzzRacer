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

#define MODE_CC 1
#define MODE_NOCC 2
#define MODE_ZERO_REF 3
#define MODE_REF 4

#define PARAM_LR 0.036
#define PARAM_L 0.09

#define TEMPERATURE %(TEMPERATURE)s
#define DT %(DT)s

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
float evaluate_step_cost( float* state, float* u, float in_raceline[][RACELINE_DIM], int* last_u);
__device__
float evaluate_terminal_cost( float* state,float* x0, float in_raceline[][RACELINE_DIM]);
__device__
float evaluate_collision_cost( float* state, float* opponent_pos);
__device__
void find_closest_id(float* state, float in_raceline[][RACELINE_DIM], int guess, int range, int* ret_idx, float* ret_dist);

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
__device__
void forward_kinematics( float* x, float* u);


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
    forward_kinematics(x, u);

    // evaluate step cost (crosstrack error and velocity deviation)
    float step_cost = evaluate_step_cost(x,u,in_raceline,&last_u);
    // cost related to collision avoidance / opponent avoidance
    // TODO too conservative
    if (i < HORIZON/2){
      for (int j=0; j<opponent_count; j++){
        for (int k=0; k<HORIZON/2; k++){ 
          //cost += evaluate_collision_cost(x,opponents_prediction[j][i]);
          // NOTE use current position only
          float this_collision_cost = evaluate_collision_cost(x,opponents_prediction[j][k]);
          cost += this_collision_cost;
          total_collision_cost += this_collision_cost;
        }
      }
    }

    int temp_index;
    float temp_dist;
    find_closest_id(x,in_raceline,last_u,RACELINE_SEARCH_RANGE, &temp_index, &temp_dist);

    cost += step_cost;

    // evaluate track boundary cost
    int u_estimate = -1;
    cost += evaluate_boundary_cost(x,x0,in_raceline, &u_estimate);

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
float evaluate_step_cost( float* state, float* u, float in_raceline[][RACELINE_DIM], int* last_u){
  //float heading = state[3];
  int idx;
  float dist;

  find_closest_id(state,in_raceline,*last_u,RACELINE_SEARCH_RANGE, &idx,&dist);
  // update estimate of closest index on raceline
  *last_u = idx;

  // heading cost
  //float cost = dist*0.5 + fabsf(fmodf(in_raceline[idx][2] - heading + PI,2*PI) - PI);

  // velocity cost
  // current FORWARD velocity - target velocity at closest ref point

  // forward vel

  float dv = state[2] - in_raceline[idx][3];
  float cost = dist + 0.1*dv*dv;
  //float cost = 2.0* dist + 0.5*dv*dv;
  //float cost = dist;
  // additional penalty on negative velocity 
  if (state[2] < 0){
    cost += 0.1;
  }
  return cost*5.0;
}

__device__
float evaluate_terminal_cost( float* state,float* x0, float in_raceline[][RACELINE_DIM]){
  //int idx0,idx;
  //float dist;

  // we don't need distance info for initial state, 
  //dist is put in as a dummy variable, it is immediately overritten
  //find_closest_id(x0,in_raceline,-1,0,&idx0,&dist);
  //find_closest_id(state,in_raceline,-1,0,&idx,&dist);

  // wrapping
  // *0.01: convert index difference into length difference
  // length of raceline is roughly 10m, with 1000 points roughly 1d_index=0.01m
  //return -1.0*float((idx - idx0 + RACELINE_LEN) %% RACELINE_LEN)*0.01;
  // NOTE ignoring terminal cost
  return 0.0;
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
    cost = (dist +0.05> in_raceline[idx][4])? 0.3:0.0;
  } else {
    cost = (dist +0.05> in_raceline[idx][5])? 0.3:0.0;
  }

  return cost;
}

__device__
void calc_feedback_control(float* controls, float*state, int index){

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
  float dvelocity = throttle * DT;
  float dpsi = velocity / PARAM_LR * sinf(beta) * DT;

  state[0] += dx;
  state[1] += dy;
  state[2] += dvelocity;
  state[3] += dpsi;

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

