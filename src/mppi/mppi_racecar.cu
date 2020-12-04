// cuda code for MPPI with dynamic bicycle model
// IMPORTANT make sure the macro declarations are accurate

#define SAMPLE_COUNT 8192
#define HORIZON 30

#define CONTROL_DIM 2
#define STATE_DIM 6
#define RACELINE_LEN 1024

#define Caf (5*0.25*0.1667*9.81)
#define Car (5*0.25*0.1667*9.81)
#define Lf (0.09-0.036)
#define Lr (0.036)
#define Iz (0.1667/12.0*(0.1*0.1+0.1*0.1))
#define Mass (0.1667)


#define TEMPERATURE 1
#define DT 0.03

#define PI 3.141592654f



// evaluate step cost based on target state x_goal,current state x and control u
// in_raceline is 2d array of size (RACELINE_LEN,3), the first dimension denote different control points, the second denote data, 0:x, 1:y, 2:heading(radian)
__device__
float evaluate_step_cost( float* state, float* u, float in_raceline[][3]);
__device__
float evaluate_terminal_cost( float* state, float* u,float* x0, float in_raceline[][3]);

// forward dynamics by one step
__device__
void forward_dynamics( float* x, float* u);

__global__
void evaluate_control_sequence(float* out_cost,float* x0, float* in_control, float* in_epsilon, float in_raceline[][3]){
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

  float* u = in_control + id*HORIZON*CONTROL_DIM; 

  // initialize cost
  float cost = 0;
  // run simulation
  for (int i=0; i<HORIZON; i++){
    // step forward dynamics, update state x in place
    forward_dynamics(x,u);

    // evaluate step cost
    cost += evaluate_step_cost(x,u,in_raceline);
    // FIXME ignoring epsilon induced cost
    /*
    for (int j=0; j<CONTROL_DIM; j++){
      cost += u[i]*in_epsilon[id*HORIZON*CONTROL_DIM + i*CONTROL_DIM + j];
    }
    */

    u += CONTROL_DIM;

  }
  cost += evaluate_terminal_cost(x,u,x0,in_raceline);
  out_cost[id] = cost;

}


__device__
void find_closest_id(float* state, float in_raceline[][3], int* ret_idx, float* ret_dist){
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
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  *ret_dist = sqrtf(current_min);
  return;

}

__device__
float evaluate_step_cost( float* state, float* u, float in_raceline[][3]){
  float heading = state[4];
  int idx;
  float dist;

  find_closest_id(state,in_raceline,&idx,&dist);
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  // heading cost
  //float cost = dist*0.5 + fabsf(fmodf(in_raceline[idx][2] - heading + PI,2*PI) - PI);
  float cost = dist*0.5;
  return cost*10.0;
}

__device__
float evaluate_terminal_cost( float* state, float* u,float* x0, float in_raceline[][3]){
  int idx0,idx;
  float dist;

  // we don't need distance info for initial state, 
  //dist is put in as a dummy variable, it is immediately overritten
  find_closest_id(x0,in_raceline,&idx0,&dist);
  find_closest_id(state,in_raceline,&idx,&dist);

  // wrapping
  // *0.01: convert index difference into length difference
  // length of raceline is roughly 10m, with 1000 points roughly 1d_index=0.01m
  //return -10.0*float((idx - idx0 + RACELINE_LEN)%(RACELINE_LEN))*0.01;
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

  float debug = steering;

  int id = blockIdx.x * blockDim.x + threadIdx.x;

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
