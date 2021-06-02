// cuda code for CC-MPPI with kinematic bicycle model 

#include <curand_kernel.h>
#define SAMPLE_COUNT %(SAMPLE_COUNT)s
#define HORIZON %(HORIZON)s

#define CONTROL_DIM %(CONTROL_DIM)s
#define STATE_DIM %(STATE_DIM)s
#define RACELINE_LEN %(RACELINE_LEN)s
#define CURAND_KERNEL_N %(CURAND_KERNEL_N)s

#define PARAM_LF (0.09-0.036)
#define PARAM_LR 0.036
#define PARAM_L 0.09
#define PARAM_DF  3.93731
#define PARAM_DR  6.23597
#define PARAM_C  2.80646
#define PARAM_B  0.51943
#define PARAM_CM1  6.03154
#define PARAM_CM2  0.96769
#define PARAM_CR  (-0.20375)
#define PARAM_CD  0.00000
#define PARAM_IZ  0.00278
#define PARAM_MASS  0.1667


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

// cost for going off track
__device__
float evaluate_boundary_cost( float* state, float* x0, float in_raceline[][RACELINE_DIM], int* u_estimate);

// forward kinematics model by one step
__device__
void forward_kinematics( float* x, float* u);

extern "C" {

// out(m*p) = A(m*n) @ B(n*p), A matrix start from A+offset*n*m
// A is assumed to be a stack of 2d matrix, offset instructs which 2d matrix to use
__device__
void matrix_multiply_helper( float* A, int offset, float* B, int m, int n, int p, float* out){
  for (int i=0; i<m; i++){
    for (int j=0; j<p; j++){
      out[i*p + j] = 0;
      for (int k=0; k<n; k++){
        out[i*p + j] += A[offset*n*m + i*n + j];
      }
    }
  }

}

// evaluate step cost based on target state x_goal,current state x and control u
// in_ref_control: dim horizon*control_dim
// in_epsilon: dim samples*horizon*control_dim, will be updated so that in_ref_control + in_epsilon respects limits
// in_raceline is 2d array of size (RACELINE_LEN,4), the first dimension denote different control points, the second denote data, 0:x, 1:y, 2:heading(radian), 3:ref velocity
__global__
void evaluate_control_sequence(
    float* out_cost,
    float* out_control,
    float* x0, 
    float* in_ref_control, 
    float* limits, 
    float* in_epsilon, 
    float in_raceline[][RACELINE_DIM], 
    float* Ks, float* As, float* Bs){

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
  for (int i=0; i<HORIZON; i++){
    float _u[CONTROL_DIM];
    float* u = _u;

    // calculate control
    // u = v(ref control) + K * y(cc feedback) + epsilon (noise)
    // feedback
    matrix_multiply_helper(Ks, i, y, CONTROL_DIM, STATE_DIM, 1, u);

    for (int j=0; j<CONTROL_DIM; j++){
      // calculate control
      float val = u[j] + in_ref_control[i*CONTROL_DIM + j] + in_epsilon[id*HORIZON*CONTROL_DIM + i*CONTROL_DIM + j];

      // apply constrain on control input
      val = val < _limits[j*CONTROL_DIM]? _limits[j*CONTROL_DIM]:val;
      val = val > _limits[j*CONTROL_DIM+1]? _limits[j*CONTROL_DIM+1]:val;
      // update epsilon
      in_epsilon[id*HORIZON*CONTROL_DIM + i*CONTROL_DIM + j] = val - in_ref_control[i*CONTROL_DIM + j];

      // set control
      u[j] = val;
      out_control[id*HORIZON*CONTROL_DIM + i*CONTROL_DIM + j] = u[j];

    }

    // step forward dynamics, update state x in place
    forward_kinematics(x, u);

    // evaluate step cost (crosstrack error and velocity deviation)
    cost += evaluate_step_cost(x,u,in_raceline,&last_u);

    // evaluate track boundary cost
    int u_estimate = -1;
    for (int k=0; k<HORIZON; k++){
      cost += evaluate_boundary_cost(x,x0,in_raceline, &u_estimate);
    }

    // FIXME ignoring epsilon induced cost
    /*
    for (int j=0; j<CONTROL_DIM; j++){
      cost += u[i]*in_epsilon[id*HORIZON*CONTROL_DIM + i*CONTROL_DIM + j];
    }
    */

    // update y
    // y = A * y + B * epsilon
    float temp[STATE_DIM];
    float temp2[STATE_DIM];
    float* this_epsilon = in_epsilon + id*HORIZON*CONTROL_DIM + i*CONTROL_DIM;

    matrix_multiply_helper(As, i, y, STATE_DIM, STATE_DIM, 1, temp);
    matrix_multiply_helper(Bs, i, this_epsilon, STATE_DIM, STATE_DIM, 1, temp2);
    for (int k=0; k<STATE_DIM; k++){
      y[k] = temp[k] + temp2[k];
    }


    u += CONTROL_DIM;

  }
  float terminal_cost = evaluate_terminal_cost(x,x0,in_raceline);
  // DEBUG
  if (id==0){
    //printf("terminal: %%.2f\n",terminal_cost/cost);
    //printf("terminal: %%.2f\n",terminal_cost);
  }
  //cost += evaluate_terminal_cost(x,x0,in_raceline);
  cost += terminal_cost;
  out_cost[id] = cost;

}

//extern c
}


// find closest id in the index range (guess - range, guess + range)
// if guess is -1 then the entire spectrum will be searched
__device__
void find_closest_id(float* state, float in_raceline[][RACELINE_DIM], int guess, int range, int* ret_idx, float* ret_dist){
  float x = state[0];
  float y = state[2];
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
  //float heading = state[4];
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
  float vx = state[1]*cosf(state[4]) + state[3]*sinf(state[4]);

  float dv = vx - in_raceline[idx][3];
  float cost = dist + 0.1*dv*dv;
  //float cost = dist;
  // additional penalty on negative velocity 
  if (vx < 0){
    cost += 0.1;
  }
  return cost;
}

__device__
float evaluate_collision_cost( float* state, float* opponent_pos){
  //float heading = state[4];

  float dx = state[0]-opponent_pos[0];
  float dy = state[2]-opponent_pos[1];
  //float cost = 2.0*(0.15 - sqrtf(dx*dx + dy*dy));
  float cost = (0.15-sqrtf(dx*dx + dy*dy)) > 0? 0.4:0;


  return cost;
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
  float raceline_to_point_angle = atan2f(in_raceline[idx][1] - state[2], in_raceline[idx][0] - state[0]) ;
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
__device__
void forward_kinematics(float* state, float* u){
  float throttle,steering;
  float x,y,velocity, psi;

  x = state[0];
  y = state[1];
  velocity = state[2]
  psi = state[3]

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

