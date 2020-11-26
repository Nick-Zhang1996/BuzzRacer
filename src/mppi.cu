// cuda code for MPPI

#define TEMPERATURE 1
#define CONTROL_DIM 2
#define SAMPLE_COUNT 10
#define STATE_DIM 4
#define HORIZON 40

__global__
void evaluate_control_sequence(float *out_cost, float *x0, float ***in_control, float ***in_epsilon){
  // get global thread id
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id>SAMPLE_COUNT){
    return;
  }
  float x[STATE_DIM];
  // copy to local state
  for (int i=0; i<STATE_DIM; i++){
    x[i] = x0[i];
  }

  // prepare constants
  const float m1 = 1;
  const float m2 = 1;
  const float k1 = 1;
  const float k2 = 1;
  const float c1 = 1.4;
  const float c2 = 1.4;
  const float dt = 0.1;

  // initialize cost
  //out_cost[id] = 0;
  float cost = 0;
  float** control = in_control[id];

  // run simulation
  for (int i=0; i<HORIZON; i++){
    float* u = control[i];

    // step forward dynamics, update state x
    float x1 = x[0];
    float dx1 = x[1];
    float x2 = x[2];
    float dx2 = x[3];

    float ddx1 = -(k1*x1 + c1*dx1 + k2*(x1-x2) + c2*(dx1-dx2)-u[0])/m1;
    float ddx2 = -(k2*(x2-x1) + c2*(dx2-dx1)-u[1])/m2;

    x1 += dx1*dt;
    dx1 += ddx1*dt;
    x2 += dx2*dt;
    dx2 += ddx2*dt;

    x[0] = x1;
    x[1] = dx1;
    x[2] = x2;
    x[3] = dx2;

    // evaluate cost, update cost
    cost += (x[0]-1)*(x[0]-1)*1.0 + x[1]*x[1]*0.1 + (x[2]-3)*(x[2]-3)*1.0 + x[3]*x[3]*0.1;
    cost += u[0]*in_epsilon[id][i][0] + u[1]*in_epsilon[id][i][1];
  }
  out_cost[id] = cost;

}
