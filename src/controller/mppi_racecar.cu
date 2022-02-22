// cuda code for MPPI with dynamic bicycle model

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

#define STATE_X 0
#define STATE_Y 1
#define STATE_HEADING 2
#define STATE_VX 3
#define STATE_VY 4
#define STATE_OMEGA 5

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
void find_closest_id(float* state, int guess, int* ret_idx, float* ret_dist);
__device__
float evaluate_boundary_cost( float* state, int* u_estimate);
__device__
float evaluate_step_cost( float* state, float* u,int* last_u);
__device__
void forward_dynamics( float* state, float* u);

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
__global__ void set_control_limit(float* in_control_limit){
  for(int i=0;i<sizeof(control_limit);i++){ control_limit[i] = in_control_limit[i];}
}
__global__ void set_noise_cov(float* in_noise_cov){
  for(int i=0;i<sizeof(noise_std);i++){ noise_std[i] = sqrtf(in_noise_cov[i]);}
  //printf("cov: %%.2f, %%.2f \n",noise_std[0],noise_std[1]);
}
__global__ void set_noise_mean(float* in_noise_mean){
  for(int i=0;i<sizeof(noise_mean);i++){ noise_mean[i] = sqrtf(in_noise_mean[i]);}
}

__global__ void set_raceline(float* in_raceline){
  for(int i=0;i<RACELINE_LEN;i++){ 
    raceline[i][0] = in_raceline[i*RACELINE_DIM + 0];
    raceline[i][1] = in_raceline[i*RACELINE_DIM + 1];
    raceline[i][2] = in_raceline[i*RACELINE_DIM + 2];
    raceline[i][3] = in_raceline[i*RACELINE_DIM + 3];
    raceline[i][4] = in_raceline[i*RACELINE_DIM + 4];
    raceline[i][5] = in_raceline[i*RACELINE_DIM + 5];
  }
}

__global__ void generate_control_noise(){
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  
  // failsafe, should never be true
  if (id >= CURAND_KERNEL_N) {return;}

  float _scales[CONTROL_DIM*2];
  for (int i=0; i<sizeof(_scales); i++){ _scales[i] = noise_std[i];}

  curandState_t s = *curand_states[id];
  int start = id*(SAMPLE_COUNT*HORIZON*CONTROL_DIM)/CURAND_KERNEL_N;
  int end = min(SAMPLE_COUNT*HORIZON*CONTROL_DIM,(id+1)*(SAMPLE_COUNT*HORIZON*CONTROL_DIM)/CURAND_KERNEL_N);
  //printf("id %%d, %%d - %%d\n",id, start, end);

  for(int i=start; i < end; i+=CONTROL_DIM ) {
    for (int j=0; j<CONTROL_DIM; j++){
      float val = curand_normal(&s) * _scales[j] + noise_mean[j];
      sampled_noise[i+j] = val;
      // DEBUG
      //out_values[i+j] = val;
    }
  }
  *curand_states[id] = s;

}
// evaluate sampled control sequences
// x0: x,y,heading, v_forward, v_sideways, omega
// ref_control: samples*horizon*control_dim
// out_cost: samples 
__global__ void evaluate_control_sequence(float* in_x0, float* ref_control, float* out_cost, float* out_control, float* out_trajectories){
  // get global thread id
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id>=SAMPLE_COUNT){
    return;
  }

  float x[STATE_DIM];
  // copy to local state
  // NOTE possible time saving by copy to local memory
  for (int i=0; i<STATE_DIM; i++){
    x[i] = *(in_x0 + i);
  }

  // initialize cost
  float cost = 0;
  // used as estimate to find closest index on raceline
  int last_u = -1;
  // run simulation
  // loop over time horizon
  for (int i=0; i<HORIZON; i++){
    float _u[CONTROL_DIM];
    float* u = _u;

    // apply constrain on control input
    for (int j=0; j<CONTROL_DIM; j++){
      float val = ref_control[i*CONTROL_DIM + j] + sampled_noise[id*HORIZON*CONTROL_DIM + i*CONTROL_DIM + j];
      val = val < control_limit[j*CONTROL_DIM]? control_limit[j*CONTROL_DIM]:val;
      val = val > control_limit[j*CONTROL_DIM+1]? control_limit[j*CONTROL_DIM+1]:val;
      out_control[id*HORIZON*CONTROL_DIM + i*CONTROL_DIM + j] = val;
      u[j] = val;
    }

    // step forward dynamics, update state x in place
    forward_dynamics(x,u);
    for (int j=0; j<STATE_DIM; j++){
      out_trajectories[id*HORIZON*STATE_DIM + i*STATE_DIM + j] = x[j];
    }

    // evaluate step cost
    cost += evaluate_step_cost(x,u,&last_u);
    cost += evaluate_boundary_cost(x,&last_u);

    u += CONTROL_DIM;

  }
  float terminal_cost = evaluate_terminal_cost(x,in_x0);
  cost += evaluate_terminal_cost(x,in_x0);
  cost += terminal_cost;
  out_cost[id] = cost;
}

//extern c
}

// x: x,y,heading, v_forward, v_sideways, omega
// u: throttle, steering1
__device__
void forward_dynamics( float* state, float* u){
  float x,vx,y,vy,heading,omega,vxg,vyg;
  float d_vx,d_vy,d_omega,slip_f,slip_r,Ffy,Fry;
  float throttle,steering;
  x = state[STATE_X];
  y = state[STATE_Y];
  heading = state[STATE_HEADING];
  vx = state[STATE_VX];
  vy = state[STATE_VY];
  omega = state[STATE_OMEGA];

  throttle = u[CONTROL_THROTTLE];
  steering = u[CONTROL_STEERING];

  // for small velocity, use kinematic model 
  if (vx<0.05){
    float beta = atanf(PARAM_LR/PARAM_L*tanf(steering));
    // motor model
    d_vx = 0.425*(15.2*throttle - vx - 3.157);
    vx = vx + d_vx * DT;
    vy = sqrtf(vx*vx + vy*vy) * sinf(beta);
    d_omega = 0.0;
    omega = vx/PARAM_L*tanf(steering);

    slip_f = 0.0;
    slip_r = 0.0;
    Ffy = 0.0;
    Fry = 0.0;

  } else {
    // dynamic model
    slip_f = -atanf((omega*PARAM_LF + vy)/vx) + steering;
    slip_r = atanf((omega*PARAM_LR - vy)/vx);

    Ffy = PARAM_DF * sinf( PARAM_C * atanf(PARAM_B *slip_f)) * 9.8 * PARAM_LR / (PARAM_LR + PARAM_LF) * PARAM_MASS;
    Fry = PARAM_DR * sinf( PARAM_C * atanf(PARAM_B *slip_r)) * 9.8 * PARAM_LF / (PARAM_LR + PARAM_LF) * PARAM_MASS;

    // motor model

    // Dynamics
    //d_vx = 1.0/PARAM_MASS * (Frx - Ffy * sinf( steering ) + PARAM_MASS * vy * omega);
    d_vx = 1.8*0.425*(15.2*throttle - vx - 3.157);
    d_vy = 1.0/PARAM_MASS * (Fry + Ffy * cosf( steering ) - PARAM_MASS * vx * omega);
    d_omega = 1.0/PARAM_IZ * (Ffy * PARAM_LF * cosf( steering ) - Fry * PARAM_LR);

    // discretization
    vx = vx + d_vx * DT;
    vy = vy + d_vy * DT;
    omega = omega + d_omega * DT ;
  }

  // back to global frame
  vxg = vx*cosf(heading)-vy*sinf(heading);
  vyg = vx*sinf(heading)+vy*cosf(heading);

  // apply updates
  x += vxg*DT;
  y += vyg*DT;
  heading += omega*DT + 0.5* d_omega * DT * DT;

  state[0] = x;
  state[1] = y;
  state[2] = heading;
  state[3] = vx;
  state[4] = vy;
  state[5] = omega;
  return;

}

__device__
float evaluate_step_cost( float* state, float* u,int* last_u){
  //float heading = state[4];
  int idx;
  float dist;

  find_closest_id(state,*last_u, &idx,&dist);
  // update estimate of closest index on raceline
  *last_u = idx;

  // heading cost
  //float cost = dist*0.5 + fabsf(fmodf(raceline[idx][2] - heading + PI,2*PI) - PI);

  // velocity cost
  // current FORWARD velocity - target velocity at closest ref point

  // forward vel
  float vx = state[STATE_VX];

  float dv = vx - raceline[idx][3];
  float cost = dist + 0.1*dv*dv;
  //float cost = dist;
  // additional penalty on negative velocity 
  if (vx < 0){
    cost += 0.1;
  }
  return cost;
}

// NOTE potential improvement by reusing idx result from other functions
// u_estimate is the estimate of index on raceline that's closest to state
__device__
float evaluate_boundary_cost( float* state,  int* u_estimate){
  int idx;
  float dist;

  // performance barrier FIXME
  find_closest_id(state,*u_estimate,  &idx,&dist);
  *u_estimate = idx;
  
  float tangent_angle = raceline[idx][4];
  float raceline_to_point_angle = atan2f(raceline[idx][1] - state[STATE_Y], raceline[idx][0] - state[STATE_X]) ;
  float angle_diff = fmodf(raceline_to_point_angle - tangent_angle + PI, 2*PI) - PI;

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
void find_closest_id(float* state, int guess, int* ret_idx, float* ret_dist){
  float x = state[STATE_X];
  float y = state[STATE_Y];
  float val;

  int idx = 0;
  float current_min = 1e6;

  int start, end;
  if (guess == -1){
    start = 0;
    end = RACELINE_LEN;
  } else {
    start = guess - RACELINE_SEARCH_RANGE;
    end = guess + RACELINE_SEARCH_RANGE;
  }

  for (int k=start;k<end;k++){
    int i = (k + RACELINE_LEN) %% RACELINE_LEN;
    val = (x-raceline[i][RACELINE_X])*(x-raceline[i][RACELINE_X]) + (y-raceline[i][RACELINE_Y])*(y-raceline[i][RACELINE_Y]);
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
