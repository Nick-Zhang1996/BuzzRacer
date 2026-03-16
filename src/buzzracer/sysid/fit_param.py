""" Fit parameter in dynamics model """
import torch
import torch.optim as optim

class DifferentiableBicycleModel:
    def __init__(self, dt=0.01):
        self.dt = dt
        # Define learnable parameters (e.g., wheelbase L)
        # Initializing with a guess of 2.5 meters
        self.L = torch.nn.Parameter(torch.tensor(2.5, requires_grad=True))
        self.params = [self.L]

    def step(self, state, control):
        """
        State: [x, y, theta, v_f, v_s] (Simplified for example)
        Control: [accel, steer]
        """
        x, y, theta, v_f, v_s = state[..., 0], state[..., 1], state[..., 2], state[..., 3], state[..., 4]
        accel, steer = control[..., 0], control[..., 1]

        # Kinematic Equations
        new_x = x + v_f * torch.cos(theta) * self.dt
        new_y = y + v_f * torch.sin(theta) * self.dt
        new_theta = theta + (v_f / self.L) * torch.tan(steer) * self.dt
        new_vf = v_f + accel * self.dt
        new_vs = v_s # Assuming kinematic, v_sideways might be 0 or simplified
        
        return torch.stack([new_x, new_y, new_theta, new_vf, new_vs], dim=-1)

# 1. Setup Data (Mocking your (N, 8) vector)
# Order: x, y, theta, v_f, v_s, steer, accel, padding/unused
N = 1000
data = torch.randn(N, 8) 
dt = 0.01
T = 10 # Prediction horizon
batch_size = 32

model = DifferentiableBicycleModel(dt=dt)
optimizer = optim.Adam(model.params, lr=0.01)

# 2. Training Loop
for epoch in range(100):
    optimizer.zero_grad()
    total_loss = 0
    
    # Randomly sample starting points t0
    indices = torch.randint(0, N - T - 1, (batch_size,))
    
    for idx in indices:
        # Initial state at t0
        current_state = data[idx, :5] # x, y, theta, v_f, v_s
        
        step_loss = 0
        for t in range(T):
            # Get control from data at current step
            control = data[idx + t, 5:7] 
            # Get ground truth state for next step
            target_state = data[idx + t + 1, :5]
            
            # Predict next state
            current_state = model.step(current_state, control)
            
            # Sum MSE Loss (you may want to weight theta differently)
            step_loss += torch.mean((current_state - target_state)**2)
        
        total_loss += step_loss / T

    total_loss /= batch_size
    total_loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {total_loss.item():.6f} | L: {model.L.item():.4f}")
