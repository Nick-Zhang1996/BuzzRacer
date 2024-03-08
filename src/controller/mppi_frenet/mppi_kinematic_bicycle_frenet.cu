// cuda code for MPPI with kinematic bicycle model in frenet coordinates

#include <curand_kernel.h>
#define SAMPLE_COUNT %(SAMPLE_COUNT)s
#define HORIZON %(HORIZON)s

#define CONTROL_DIM %(CONTROL_DIM)s
#define STATE_DIM %(STATE_DIM)s
// size of discretized raceline
#define RACELINE_LEN %(RACELINE_LEN)s
// curve length of raceline in meter
#define RACELINE_LEN_M %(RACELINE_LEN_M)s
#define CURAND_KERNEL_N %(CURAND_KERNEL_N)s

#define OBSTACLE_RADIUS 0.1

#define PARAM_LF 0.04824
#define PARAM_LR (0.09-0.04824)
#define PARAM_L 0.09

#define TEMPERATURE %(TEMPERATURE)s
#define DT %(DT)s

#define PI 3.141592654f

#define RACELINE_DIM 5
#define RACELINE_S 0
#define RACELINE_CURVATURE 1
#define RACELINE_V 2
#define RACELINE_LEFT_BOUNDARY 3
#define RACELINE_RIGHT_BOUNDARY 4

#define STATE_S 0
#define STATE_V 1
#define STATE_N 2
#define STATE_PHI 3
#define STATE_BETA 4

#define CONTROL_STEERING 0
#define CONTROL_THROTTLE 1

// one discretization step is around 1cm
#define RACELINE_SEARCH_RANGE 10

// parameters to define cost
#define COST_Q_S %(COST_Q_S)s
#define COST_Q_V %(COST_Q_V)s
#define COST_Q_N %(COST_Q_N)s
#define COST_Q_PHI %(COST_Q_PHI)s

#define COST_q_S %(COST_q_S)s
#define COST_q_V %(COST_q_V)s
#define COST_q_N %(COST_q_N)s
#define COST_q_PHI %(COST_q_PHI)s

#define COST_QOP_1 %(COST_QOP_1)s
#define COST_QOP_2 %(COST_QOP_2)s

#define COST_BDRY_MIN %(COST_BDRY_MIN)s
#define COST_BDRY %(COST_BDRY)s

#define COST_OPPO_MIN_S %(COST_OPPO_MIN_S)s
#define COST_OPPO_MIN_N %(COST_OPPO_MIN_N)s
#define COST_Q_COL %(COST_Q_COL)s

#define PARAM_MAX_AX %(PARAM_MAX_AX)s
#define PARAM_MAX_AY %(PARAM_MAX_AY)s
#define PARAM_MAX_V %(PARAM_MAX_V)s


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
float evaluate_boundary_cost( float* state, int* u_estimate);
__device__
float evaluate_step_cost( float* state, float* u);
__device__
float evaluate_collision_cost( float* state, int step, float* opponent_traj,int opponent_id);
__device__
void forward_dynamics( float* state, float* u, int* last_index);
__device__
float tire_curve( float slip);
__device__
void get_curvature(float* state, int* io_idx, float* o_curvature);
__device__
float map(float val, float in_l,float in_h,float out_low,float out_high);
__device__
float sqrf(float val){ return val*val; }
__device__
void bound_control(float* state, float* i_control, float* o_control);

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
    for (int j=0;j<RACELINE_DIM;j++){
      raceline[i][j] = in_raceline[i*RACELINE_DIM + j];
    }
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
// x0: s,v,n,phi,beta
// u0: current control to penalize control time rate
// ref_control: samples*horizon*control_dim
// out_cost: samples 
// out_trajectories: output trajectories, samples*horizon*n
// opponent_count: integer
// opponent_traj: opponent_count * prediction_horizon * (s,v,n,phi,beta)
//__global__ void evaluate_control_sequence(float* in_x0, float* in_u0, float* ref_dudt, float* out_cost, float* out_dudt, int opponent_count, float* in_opponent_traj, float* out_trajectories){
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
  // copy to local register gives faster performance
  float last_u[CONTROL_DIM];
  for (int i=0; i<CONTROL_DIM; i++){
    last_u[i] = *(in_u0+i);
  }

  // run simulation
  // loop over time horizon
  for (int i=0; i<HORIZON; i++){
    float _u[CONTROL_DIM];
    float* u = _u;

    // apply constrain on control input
    for (int j=0; j<CONTROL_DIM; j++){
      float dudt = (ref_dudt[i*CONTROL_DIM + j] + sampled_noise[id*HORIZON*CONTROL_DIM + i*CONTROL_DIM + j]);
      u[j] = last_u[j] + dudt * DT;
    }
    bound_control(x,u,u);
    for (int j=0; j<CONTROL_DIM; j++){
      out_dudt[id*HORIZON*CONTROL_DIM + i*CONTROL_DIM + j] = (u[j]-last_u[j])/DT;
    }


    // update output trajectories
    /*
    // DEBUG output sampled trajectory
    for (int j=0; j<STATE_DIM; j++){
      out_trajectories[id*HORIZON*STATE_DIM + i*STATE_DIM + j] = x[j];
    }
    */

    // step forward dynamics, update state x in place
    forward_dynamics(x,u,&last_index);

    // evaluate step cost
    cost += evaluate_step_cost(x, u);
    cost += evaluate_boundary_cost(x,&last_index);
    for (int k=0;k<opponent_count;k++){
      cost += evaluate_collision_cost(x,i,in_opponent_traj,k);
    }

    for (int k=0; k<CONTROL_DIM; k++){
      last_u[k] = u[k];
    }
  }
  float terminal_cost = evaluate_terminal_cost(x,in_x0, &last_index);
  cost += terminal_cost;
  out_cost[id] = cost;
}

//extern c
}

// x: s,v,n,phi,beta
// u: steering, throttle
__device__
void forward_dynamics( float* state, float* u, int* last_index){
  float s,v,n,phi,beta,ax,ay;
  float k_s,dsdt,dvdt,dndt,dbetadt,dphidt;
  s = state[STATE_S];
  v = state[STATE_V];
  n = state[STATE_N];
  phi = state[STATE_PHI];
  beta = state[STATE_BETA];
  ax = u[CONTROL_THROTTLE];
  ay = u[CONTROL_STEERING];

  get_curvature(state, last_index,&k_s);
  dsdt = v*cosf(phi)/(1-n*k_s);
  dvdt = cosf(beta)*ax + sinf(beta)*ay;
  dndt = v*sinf(phi);
  dbetadt = (-sinf(beta)*ax + cosf(beta) *ay)/v;
  dphidt = dbetadt + v/PARAM_LR*sinf(beta)-v*cosf(phi)*k_s/(1-n*k_s);

  // apply updates
  s += dsdt*DT;
  v += dvdt*DT;
  n += dndt*DT;
  beta += dbetadt*DT;
  phi += dphidt*DT;

  state[STATE_S] = s;
  state[STATE_V] = v;
  state[STATE_N] = n;
  state[STATE_PHI] = phi;
  state[STATE_BETA] = beta;
  return;

}

__device__
float evaluate_step_cost( float* state, float* u){
  float cost = 0.0;
  cost += 0.5*COST_Q_S *state[STATE_S]*state[STATE_S];
  cost += 0.5*COST_Q_V *state[STATE_V]*state[STATE_V];
  cost += 0.5*COST_Q_N *state[STATE_N]*state[STATE_N];
  cost += 0.5*COST_Q_PHI *state[STATE_PHI]*state[STATE_PHI];

  cost += COST_q_S *state[STATE_S];
  cost += COST_q_V *state[STATE_V];
  cost += COST_q_N *state[STATE_N];
  cost += COST_q_PHI *state[STATE_PHI];
  // TODO add control cost

  return cost;
}
// NOTE potential improvement by reusing idx result from other functions
// u_estimate is the estimate of index on raceline that's closest to state
__device__
float evaluate_boundary_cost( float* state,  int* last_index){
  float cost = 0.0;
  const float left = max(raceline[*last_index][RACELINE_LEFT_BOUNDARY]-COST_BDRY_MIN,0.0f);
  const float right = max(raceline[*last_index][RACELINE_RIGHT_BOUNDARY]-COST_BDRY_MIN,0.0f);
  if (state[STATE_N] > left){
    cost += COST_BDRY* sqrf(state[STATE_N] - left);
    cost += COST_BDRY;
  }

  if (-state[STATE_N] > right){
    cost += COST_BDRY* sqrf(-state[STATE_N] - right);
    cost += COST_BDRY;
  }

  return cost;
}

// find closest id in the index range (guess - range, guess + range)
// if guess is -1 then the entire spectrum will be searched
__device__
void get_curvature(float* state, int* io_idx, float* o_curvature){
  float s = fmodf(state[STATE_S],RACELINE_LEN_M);
  float val;

  int idx = 0;
  float current_min = 1e6;

  int start, end;
  if (*io_idx == -1){
    start = 0;
    end = RACELINE_LEN;
  } else {
    start = *io_idx - RACELINE_SEARCH_RANGE;
    end = *io_idx + RACELINE_SEARCH_RANGE;
  }

  for (int k=start;k<end;k++){
    int i = (k + RACELINE_LEN) %% RACELINE_LEN;
    val = (s-raceline[i][RACELINE_S])*(s-raceline[i][RACELINE_S]);
    if (val < current_min){
      idx = i;
      current_min = val;
    }
  }
  *io_idx = idx;
  *o_curvature = (s>raceline[idx][RACELINE_S])?
    map(s,raceline[idx][RACELINE_S],raceline[(idx+1)%%RACELINE_LEN][RACELINE_S],raceline[idx][RACELINE_CURVATURE],raceline[(idx+1)%%RACELINE_LEN][RACELINE_CURVATURE])
    :
    map(s,raceline[(idx-1)%%RACELINE_LEN][RACELINE_S],raceline[idx][RACELINE_S],raceline[(idx-1)%%RACELINE_LEN][RACELINE_CURVATURE],raceline[idx][RACELINE_CURVATURE]);

  return;

}

__device__
void get_curvature_debug(float* state, int* io_idx, float* o_curvature){
  float s = fmodf(state[STATE_S],RACELINE_LEN_M);
  float val;

  int idx = 0;
  float current_min = 1e6;

  int start, end;
  if (*io_idx == -1){
    start = 0;
    end = RACELINE_LEN;
  } else {
    start = *io_idx - RACELINE_SEARCH_RANGE;
    end = *io_idx + RACELINE_SEARCH_RANGE;
  }

  for (int k=start;k<end;k++){
    int i = (k + RACELINE_LEN) %% RACELINE_LEN;
    val = (s-raceline[i][RACELINE_S])*(s-raceline[i][RACELINE_S]);
    if (val < current_min){
      idx = i;
      current_min = val;
    }
  }
  *io_idx = idx;
  printf("idx: %%d, ref_s: %%.2f, s: %%.2f\n",idx,raceline[idx][RACELINE_S],s);
  *o_curvature = (s>raceline[idx][RACELINE_S])?
    map(s,raceline[idx][RACELINE_S],raceline[(idx+1)%%RACELINE_LEN][RACELINE_S],raceline[idx][RACELINE_CURVATURE],raceline[(idx+1)%%RACELINE_LEN][RACELINE_CURVATURE])
    :
    map(s,raceline[(idx-1)%%RACELINE_LEN][RACELINE_S],raceline[idx][RACELINE_S],raceline[(idx-1)%%RACELINE_LEN][RACELINE_CURVATURE],raceline[idx][RACELINE_CURVATURE]);

  return;

}

__device__
float map(float val, float in_l,float in_h,float out_low,float out_high){
  val = (val<in_l)?in_l:val;
  val = (val>in_h)?in_h:val;
  return (val-in_l)/(in_h-in_l)*(out_high-out_low)+out_low;
}

__device__
float evaluate_terminal_cost( float* current_state,float* initial_state, int* last_index ){
  return 0.0;
}

// opponent_traj: opponent_count * horizon * [x,y]
__device__
float evaluate_collision_cost( float* state, int step, float* opponent_traj, int opponent_id){

  float cost = 0.0;
  float ds = state[STATE_S] - opponent_traj[opponent_id*HORIZON*STATE_DIM + step*STATE_DIM + STATE_S];
  float dn = state[STATE_N] - opponent_traj[opponent_id*HORIZON*STATE_DIM + step*STATE_DIM + STATE_N];
  if (fabsf(ds) < COST_OPPO_MIN_S && fabsf(dn) < COST_OPPO_MIN_N){
    cost += COST_Q_COL*(ds*ds+dn*dn);
  }

  return cost;
}

__device__
void bound_control(float* state, float* i_control, float* o_control){
  float v = fabsf(state[STATE_V]);
  v = (v>PARAM_MAX_V)?PARAM_MAX_V:v;
  v = (v<0)?0:v;
  float max_acc = PARAM_MAX_AX * (1-v/PARAM_MAX_V);
  // first scale to ellipse y/aym^2+x/axm^2=1
  // then cap ax to  (-infty,max_acc]
  float ay_normalized = i_control[CONTROL_STEERING]/PARAM_MAX_AY;
  float ax_normalized = i_control[CONTROL_THROTTLE]/PARAM_MAX_AX;
  float theta = atan2f(ax_normalized,ay_normalized);
  float r = sqrtf(sqrf(ax_normalized)+sqrf(ay_normalized));
  if (r>1.0){
    r = 1.0;
  }
  o_control[CONTROL_STEERING] = PARAM_MAX_AY*r*cosf(theta);
  o_control[CONTROL_THROTTLE] = PARAM_MAX_AX*r*sinf(theta);
  o_control[CONTROL_THROTTLE] = (o_control[CONTROL_THROTTLE]>max_acc)?max_acc:o_control[CONTROL_THROTTLE];
}
