# functions needed from the rc-vip environment
from copg_optim.critic_functions import critic_update, get_advantage
import car_racing_simulator.VehicleModel as VehicleModel
import car_racing_simulator.Track as Track
from car_racing.orca_env_function import getfreezeTimecollosionReachedreward

reward1, reward2, done_c1, done_c2, coll_c1, coll_c2, counter1, counter2 = getfreezeTimecollosionReachedreward(state_c1, state_c2,
vehicle_model.getLocalBounds(state_c1[:, 0]),

# ####


# Vehicle Dynamics

# sample use
vehicle_model = VehicleModel.VehicleModel(config["n_batch"], 'cpu', config)
state_c1 = vehicle_model.dynModelBlendBatch(state_c1.view(-1,6), action1.view(-1,2)).view(-1,6)
# input: concatenated state+action type:Tensor dim: batch_n * (6+2)
# return: next state type:Tensor  dim: batch_n * (6)
# state:
