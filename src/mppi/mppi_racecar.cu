// cuda code for MPPI with dynamic bicycle model
// IMPORTANT make sure the macro declarations are accurate

#include <curand_kernel.h>
#define SAMPLE_COUNT %(SAMPLE_COUNT)s
#define HORIZON %(HORIZON)s

#define CONTROL_DIM %(CONTROL_DIM)s
#define STATE_DIM %(STATE_DIM)s
#define RACELINE_LEN %(RACELINE_LEN)s
#define CURAND_KERNEL_N %(CURAND_KERNEL_N)s

#define Caf (5*0.25*0.1667*9.81)
#define Car (5*0.25*0.1667*9.81)
#define Lf (0.09-0.036)
#define Lr (0.036)
#define Iz (0.1667/12.0*(0.1*0.1+0.1*0.1))
#define Mass (0.1667)


#define TEMPERATURE %(TEMPERATURE)s
#define DT %(DT)s

#define PI 3.141592654f



__device__
float evaluate_step_cost( float* state, float* u, float in_raceline[][4]);
__device__
float evaluate_terminal_cost( float* state,float* x0, float in_raceline[][4]);
__device__
float evaluate_collision_cost( float* state, float* opponent_pos);

// forward dynamics by one step
__device__
void forward_dynamics( float* x, float* u);

// forward kinematics model by one step
__device__
void forward_kinematics( float* x, float* u);

extern "C" {
// evaluate step cost based on target state x_goal,current state x and control u
// in_ref_control: dim horizon*control_dim
// in_epsilon: dim samples*horizon*control_dim, will be updated so that in_ref_control + in_epsilon respects limits
// in_raceline is 2d array of size (RACELINE_LEN,4), the first dimension denote different control points, the second denote data, 0:x, 1:y, 2:heading(radian), 3:ref velocity
__global__
void evaluate_control_sequence(float* out_cost,float* x0, float* in_ref_control, float* limits, float* in_epsilon, float in_raceline[][4], float opponents_prediction[][HORIZON+1][2],int opponent_count){
  // get global thread id
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id>=SAMPLE_COUNT){
    return;
  }


  float x[STATE_DIM];
  // copy to local state
  // NOTE possible time saving by copy to local memory
  for (int i=0; i<STATE_DIM; i++){
    x[i] = *(x0 + i);
  }

  float _limits[CONTROL_DIM*2];
  for (int i=0; i<CONTROL_DIM*2; i++){
    _limits[i] = limits[i];
  }

  // DEBUG
  /*
  if (id==0 && opponent_count>0){
    float dx = x[0]-opponents_prediction[0][0][0];
    float dy = x[2]-opponents_prediction[0][0][1];
    float opponent_dis = sqrtf(dx*dx + dy*dy);
    printf("dist = %% .2f \n",opponent_dis);
    //printf("x = %%.2f, y= %%.2f \n",x[0],x[2]);
  }
  */


  // initialize cost
  float cost = 0;
  // run simulation
  for (int i=0; i<HORIZON; i++){
    float _u[CONTROL_DIM];
    float* u = _u;
    for (int j=0; j<CONTROL_DIM; j++){
      float val = in_ref_control[i*CONTROL_DIM + j] + in_epsilon[id*HORIZON*CONTROL_DIM + i*CONTROL_DIM + j];
      val = val < _limits[j*CONTROL_DIM]? _limits[j*CONTROL_DIM]:val;
      val = val > _limits[j*CONTROL_DIM+1]? _limits[j*CONTROL_DIM+1]:val;
      in_epsilon[id*HORIZON*CONTROL_DIM + i*CONTROL_DIM + j] = val - in_ref_control[i*CONTROL_DIM + j];
      u[j] = val;

    }
    // step forward dynamics, update state x in place
    forward_kinematics(x,u);

    // evaluate step cost
    cost += evaluate_step_cost(x,u,in_raceline);
    // cost related to collision avoidance / opponent avoidance
    // TODO too conservative
    for (int j=0; j<opponent_count; j++){
      for (int k=0; k<HORIZON/2; k++){
      cost += evaluate_collision_cost(x,opponents_prediction[j][k]);
      }
    }


    // FIXME ignoring epsilon induced cost
    /*
    for (int j=0; j<CONTROL_DIM; j++){
      cost += u[i]*in_epsilon[id*HORIZON*CONTROL_DIM + i*CONTROL_DIM + j];
    }
    */

    u += CONTROL_DIM;

  }
  float terminal_cost = evaluate_terminal_cost(x,x0,in_raceline);
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


__device__
void find_closest_id(float* state, float in_raceline[][4], int* ret_idx, float* ret_dist){
  float x = state[0];
  float y = state[2];
  float val;

  int idx = 0;
  float current_min = 1e6;
  for (int i=0;i<RACELINE_LEN;i++){
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
float evaluate_step_cost( float* state, float* u, float in_raceline[][4]){
  //float heading = state[4];
  int idx;
  float dist;

  find_closest_id(state,in_raceline,&idx,&dist);

  // heading cost
  //float cost = dist*0.5 + fabsf(fmodf(in_raceline[idx][2] - heading + PI,2*PI) - PI);

  // velocity cost
  // current velocity - target velocity at closest ref point
  float dv = sqrtf(state[1]*state[1] + state[3]*state[3]) - in_raceline[idx][3];
  float cost = dist + 0.1*dv*dv;
  //float cost = dist;
  return cost*5.0;
}

__device__
float evaluate_collision_cost( float* state, float* opponent_pos){
  //float heading = state[4];

  float dx = state[0]-opponent_pos[0];
  float dy = state[2]-opponent_pos[1];
  float cost = 1.0*(0.1 - sqrtf(dx*dx + dy*dy))*5.0;

  return cost>0?cost:0;
}

__device__
float evaluate_terminal_cost( float* state,float* x0, float in_raceline[][4]){
  int idx0,idx;
  float dist;

  // we don't need distance info for initial state, 
  //dist is put in as a dummy variable, it is immediately overritten
  find_closest_id(x0,in_raceline,&idx0,&dist);
  find_closest_id(state,in_raceline,&idx,&dist);

  // wrapping
  // *0.01: convert index difference into length difference
  // length of raceline is roughly 10m, with 1000 points roughly 1d_index=0.01m
  //return -1.0*float((idx - idx0 + RACELINE_LEN) %% RACELINE_LEN)*0.01;
  // NOTE ignoring terminal cost
  return 0.0;
}

// update x in place
__device__
void forward_dynamics(float* state,float* u){
  float x,dx,y,dy,psi,dpsi;
  float throttle,steering;

  x = state[0];
  dx = state[1];
  y = state[2];
  dy = state[3];
  psi = state[4];
  dpsi = state[5];

  throttle = u[0];
  steering = u[1];

  x += dx * DT;
  y += dy * DT;
  psi += dpsi * DT;

  float local_dx = dx*cosf(-psi) - dy*sinf(-psi);
  float local_dy = dx*sinf(-psi) + dy*cosf(-psi);

  float d_local_dx = throttle*DT;
  float d_local_dy = (-(2*Caf+2*Car)/(Mass*local_dx)*local_dy + (-local_dx - (2*Caf*Lf-2*Car*Lr)/(Mass*local_dx)) * dpsi + 2*Caf/Mass*steering)*DT;
  float d_dpsi = (-(2*Lf*Caf - 2*Lr*Car)/(Iz*local_dx)*local_dy - (2*Lf*Lf*Caf + 2*Lr*Lr*Car)/(Iz*local_dx)*dpsi + 2*Lf*Caf/Iz*steering)*DT;

  local_dx += d_local_dx;
  local_dy += d_local_dy;
  dpsi += d_dpsi;

  // convert back to global frame
  dx = local_dx*cosf(psi) - local_dy*sinf(psi);
  dy = local_dx*sinf(psi) + local_dy*cosf(psi);

  state[0] = x;
  state[1] = dx;
  state[2] = y;
  state[3] = dy;
  state[4] = psi;
  state[5] = dpsi;

  return;
}

// forward dynamics using kinematic model
// note this model is tuned on actual car data, it may not work well with dynamic simulator
__device__
void forward_kinematics(float* state, float* u){
  float dx,dy,psi,dpsi;
  float throttle,steering;

  //x = state[0];
  dx = state[1];
  //y = state[2];
  dy = state[3];
  psi = state[4];
  //dpsi = state[5];

  throttle = u[0];
  steering = u[1];

  // convert to car frame
  float local_dx = dx*cosf(-psi) - dy*sinf(-psi);
  float local_dy = dx*sinf(-psi) + dy*cosf(-psi);

  // what if steering -> 0
  // avoid numerical instability
  //float R = 0.102/tanf(steering);
  //float beta = atanf(0.036/R);
  float beta = atanf(0.036/0.102*tanf(steering));
  local_dx += (throttle - 0.24) * 7.0 * DT;
  // avoid negative velocity
  local_dx = local_dx>0.0? local_dx:0.0;
  local_dy =  sqrtf(local_dx*local_dx + local_dy*local_dy) * sinf(beta);
  local_dy += -0.68*local_dx*steering;

  dpsi = local_dx/0.102*tanf(steering);

  // convert back to global frame
  dx = local_dx*cosf(psi) - local_dy*sinf(psi);
  dy = local_dx*sinf(psi) + local_dy*cosf(psi);

  state[0] += dx * DT;
  state[1] = dx;
  state[2] += dy * DT;
  state[3] = dy;
  state[4] += dpsi * DT;
  state[5] = dpsi;

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




// dummy placeholder main() to trick compiler into compiling when testing
/*
int main(void){
  int blockSize = 1;
  int numBlocks = 256;
  int  N = 100;
  float *out_cost,*x0,*in_control,*in_epsilon ;
  float in_raceline[][3];
   
  cudaMallocManaged(&out_cost, N*sizeof(float));
  cudaMallocManaged(&x0, N*sizeof(float));
  cudaMallocManaged(&in_control, N*sizeof(float));
  cudaMallocManaged(&in_epsilon, N*sizeof(float));
  cudaMallocManaged(&in_raceline, N*2*sizeof(float));

  evaluate_control_sequence<<<numBlocks, blockSize>>>(out_cost,x0,in_control,in_epsilon,in_raceline);
  return 0;

}
*/
