// cuda code for MPPI with dynamic bicycle model

#include <curand_kernel.h>
#define SAMPLE_COUNT %(SAMPLE_COUNT)s
#define HORIZON %(HORIZON)s

#define CONTROL_DIM %(CONTROL_DIM)s
#define STATE_DIM %(STATE_DIM)s
#define RACELINE_LEN %(RACELINE_LEN)s
#define CURAND_KERNEL_N %(CURAND_KERNEL_N)s

#define OBSTACLE_RADIUS 0.1

#define PARAM_LF 0.04824
#define PARAM_LR (0.09-0.04824)
#define PARAM_L 0.09

#define PARAM_IZ 417757e-9
#define PARAM_MASS 0.1667

#define PARAM_B 2.3
#define PARAM_C 1.6
#define PARAM_D 1.1

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
float evaluate_terminal_cost( float* current_state,float* initial_state, int* last_index);
__device__
void find_closest_id(float* state, int guess, int* ret_idx, float* ret_dist);
__device__
float evaluate_boundary_cost( float* state, int* u_estimate);
__device__
float evaluate_step_cost( float* state, float* last_u, float* u,int* last_index);
__device__
float evaluate_collision_cost( float* state, float* opponent_traj,int opponent_id);
__device__
void forward_dynamics( float* state, float* u);
__device__
float tire_curve( float slip);

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
  //printf("std: %%.2f, %%.2f \n",noise_std[0],noise_std[1]);
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
// u0: current control to penalize control time rate
// ref_control: samples*horizon*control_dim
// out_cost: samples 
// out_trajectories: output trajectories, samples*horizon*n
// opponent_count: integer
// opponent_traj: opponent_count * prediction_horizon * 2(x,y)
//__global__ void evaluate_control_sequence(float* in_x0, float* in_u0, float* ref_dudt, float* out_cost, float* out_dudt, float* out_trajectories){
__global__ void evaluate_control_sequence(float* in_x0, float* in_u0, float* ref_dudt, float* out_cost, float* out_dudt, int opponent_count, float* in_opponent_traj){
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
  int last_index = -1;
  float last_u[CONTROL_DIM];
  for (int i=0; i<CONTROL_DIM; i++){
    last_u[i] = *(in_u0+i);
  }

  /*
  if (id == 0){
    printf("last u0=%%.2f",last_u[1]*180.0/PI);
  }
  */

  // run simulation
  // loop over time horizon
  for (int i=0; i<HORIZON; i++){
    float _u[CONTROL_DIM];
    float* u = _u;

    // apply constrain on control input
    for (int j=0; j<CONTROL_DIM; j++){
      // NOTE control is variation
      float dudt = (ref_dudt[i*CONTROL_DIM + j] + sampled_noise[id*HORIZON*CONTROL_DIM + i*CONTROL_DIM + j]);
      float val = last_u[j] + dudt * DT;
      val = val < control_limit[j*CONTROL_DIM]? control_limit[j*CONTROL_DIM]:val;
      val = val > control_limit[j*CONTROL_DIM+1]? control_limit[j*CONTROL_DIM+1]:val;
      //out_dudt[id*HORIZON*CONTROL_DIM + i*CONTROL_DIM + j] = val;
      out_dudt[id*HORIZON*CONTROL_DIM + i*CONTROL_DIM + j] = (val - last_u[j])/DT;
      u[j] = val;
    }

    // step forward dynamics, update state x in place
    forward_dynamics(x,u);
    /*
    // update output trajectories
    for (int j=0; j<STATE_DIM; j++){
      out_trajectories[id*HORIZON*STATE_DIM + i*STATE_DIM + j] = x[j];
    }
    */

    // evaluate step cost
    if (i==0){
      cost += evaluate_step_cost(x, last_u, u,&last_index);
    } else {
      cost += evaluate_step_cost(x, u, u,&last_index);
    }
    cost += evaluate_boundary_cost(x,&last_index);
    for (int k=0;k<opponent_count;k++){
      cost += evaluate_collision_cost(x,in_opponent_traj,k);
    }

    for (int k=0; k<CONTROL_DIM; k++){
      last_u[k] = u[k];
    }

    u += CONTROL_DIM;

  }
  float terminal_cost = evaluate_terminal_cost(x,in_x0, &last_index);
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
    d_vx = 6.17*(throttle - vx/15.2 -0.333);
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

    //Ffy = PARAM_DF * sinf( PARAM_C * atanf(PARAM_B *slip_f)) * 9.8 * PARAM_LR / (PARAM_LR + PARAM_LF) * PARAM_MASS;
    //Fry = PARAM_DR * sinf( PARAM_C * atanf(PARAM_B *slip_r)) * 9.8 * PARAM_LF / (PARAM_LR + PARAM_LF) * PARAM_MASS;
    Ffy = 0.9*tire_curve(slip_f) * 9.8 * PARAM_LR / (PARAM_LR + PARAM_LF) * PARAM_MASS;
    Fry = tire_curve(slip_r) * 9.8 * PARAM_LF / (PARAM_LR + PARAM_LF) * PARAM_MASS;

    // motor model

    // Dynamics
    //d_vx = 1.0/PARAM_MASS * (Frx - Ffy * sinf( steering ) + PARAM_MASS * vy * omega);
    d_vx = 6.17*(throttle - vx/15.2 -0.333);
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
float evaluate_step_cost( float* state, float* last_u, float* u,int* last_index){
  //float heading = state[4];
  int idx;
  float dist;

  find_closest_id(state,*last_index, &idx,&dist);
  // update estimate of closest index on raceline
  *last_index = idx;

  // VX: current FORWARD velocity - target velocity at closest ref point
  // velocity deviation from reference velocity profile
  float dv = state[STATE_VX] - raceline[idx][RACELINE_V];

  // heading cost
  float heading_cost = fmodf(raceline[idx][RACELINE_HEADING] - state[STATE_HEADING] + 3*PI,2*PI) - PI;
  float cost = 0.1*dist*dist + 0.6*dv*dv + 0.5*heading_cost*heading_cost;
  // additional penalty on negative velocity 
  if (state[STATE_VX] < 0.05){
    cost += 0.2;
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
  
  float tangent_angle = raceline[idx][RACELINE_HEADING];
  float raceline_to_point_angle = atan2f(raceline[idx][RACELINE_Y] - state[STATE_Y], raceline[idx][0] - state[STATE_X]) ;
  float angle_diff = fmodf(raceline_to_point_angle - tangent_angle + PI, 2*PI) - PI;

  float cost;

  float coeff = 1.0;
  if (angle_diff > 0.0){
    // point is to left of raceline
    //cost = (dist +0.05> raceline[idx][4])? 0.3:0.0;
    cost = coeff*(atanf(-(raceline[idx][RACELINE_LEFT_BOUNDARY]-(dist+0.05))*100)/PI*2+1.0f);
    cost = max(0.0,cost);

  } else {
    //cost = (dist +0.05> raceline[idx][5])? 0.3:0.0;
    cost = coeff*(atanf(-(raceline[idx][RACELINE_RIGHT_BOUNDARY]-(dist+0.05))*100)/PI*2+1.0f);
    cost = max(0.0,cost);
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
float evaluate_terminal_cost( float* current_state,float* initial_state, int* last_index ){
  int idx0,idx;
  float dist;

  find_closest_id(initial_state,-1,&idx0,&dist);
  find_closest_id(current_state,*last_index,&idx,&dist);

  // wrapping
  // *0.01: convert index difference into length difference
  // length of raceline is roughly 10m, with 1000 points roughly 1d_index=0.01m
  return HORIZON*DT*4.0*2.0 -2.0*float((idx - idx0 + RACELINE_LEN) %% RACELINE_LEN)*0.01;
  //return 0.0;
}

__device__
float tire_curve( float slip){
  return PARAM_D * sinf( PARAM_C * atanf( PARAM_B * slip) );

}

// opponent_traj: opponent_count * horizon * [x,y]
__device__
float evaluate_collision_cost( float* state, float* opponent_traj, int opponent_id){

  float cost = 0.0;
  for (int i=0; i<HORIZON;i++){
    float dx = state[STATE_X] - opponent_traj[opponent_id*HORIZON*2 + i*2 + 0];
    float dy = state[STATE_Y] - opponent_traj[opponent_id*HORIZON*2 + i*2 + 1];
    float dist = sqrtf(dx*dx + dy*dy) ;
    // arctan based cost function, ramps to 2.0
    float temp = 3*(atanf(-(dist-OBSTACLE_RADIUS)*100)/PI*2+1.0f);
    cost += max(0.0,temp);
  }

  return cost;
}
