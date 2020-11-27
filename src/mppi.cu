// cuda code for MPPI

#define TEMPERATURE 1
#define CONTROL_DIM 2
#define SAMPLE_COUNT 10
#define STATE_DIM 1
#define HORIZON 20

__global__
void evaluate_control_sequence(float *out_cost, float *x0, float *in_control, float *in_epsilon){
  // get global thread id
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id>=SAMPLE_COUNT){
    return;
  }
  //printf("id = %d\n",id);
  float x[STATE_DIM];
  // copy to local state
  for (int i=0; i<STATE_DIM; i++){
    x[i] = x0[i];
  }
  //printf("id = %d, x0, %.2f, %.2f, %.2f, %.2f \n",id,x0[0],x0[1],x0[2],x0[3]);

  // prepare constants
  float m1 = 1;
  float m2 = 1;
  float k1 = 1;
  float k2 = 1;
  float c1 = 1.4;
  float c2 = 1.4;
  float dt = 0.1;
  float x1;
  float dx1;
  float x2;
  float dx2;

  // initialize cost
  //out_cost[id] = 0;
  float cost = 0;
  // run simulation
  for (int i=0; i<HORIZON; i++){
    float u0 = in_control[id*HORIZON*CONTROL_DIM + i*CONTROL_DIM];
    float u1 = in_control[id*HORIZON*CONTROL_DIM + i*CONTROL_DIM + 1];

    // step forward dynamics, update state x
    x1 = x[0];
    dx1 = x[1];
    x2 = x[2];
    dx2 = x[3];

    float ddx1 = -(k1*x1 + c1*dx1 + k2*(x1-x2) + c2*(dx1-dx2)-u0)/m1;
    float ddx2 = -(k2*(x2-x1) + c2*(dx2-dx1)-u1)/m2;

    //printf("id = %d, ddx1=%.2f, ddx2=%.2f \n",id,ddx1,ddx2);
    //float temp=-(k1*x1 + c1*dx1 + k2*(x1-x2) + c2*(dx1-dx2)-u0)/m1;
    float temp= c1*dx1;
    printf("id = %d, %.2f \n",id,temp);

    x1 += dx1*dt;
    dx1 += ddx1*dt;
    x2 += dx2*dt;
    dx2 += ddx2*dt;
    //printf("id = %d, step = %d cost = %.2f, x= %.2f, %.2f, %.2f, %.2f \n",id,i,cost,x1,dx1,x2,dx2);

    x[0] = x1;
    x[1] = dx1;
    x[2] = x2;
    x[3] = dx2;

    // evaluate cost, update cost
    cost = cost + (x[0]-1)*(x[0]-1)*1.0 + x[1]*x[1]*0.1 + (x[2]-3)*(x[2]-3)*1.0 + x[3]*x[3]*0.1;
    cost = cost + u0*in_epsilon[id*HORIZON*CONTROL_DIM + i*CONTROL_DIM] + u1*in_epsilon[id*HORIZON*CONTROL_DIM + i*CONTROL_DIM + 1];

  }
  //out_cost[id] = cost;
  // NOTE debug
  //*(out_cost+id) = id;
  out_cost[id] = cost;

  printf("id = %d, cost=%.2f \n",id,cost);

}
