# refer to paper
# The Kinematic Bicycle Model: a Consistent Model for Planning Feasible Trajectories for Autonomous Vehicles?
import numpy as np

class kinematicSimulator():

    def __init__(self,x,y,v,heading):
        # X,Y, velocity, heading
        self.states = np.array([x,y,v,heading])
        self.lf = 90e-3*0.95
        self.lr = 90e-3*0.05
        self.L = self.lf+self.lr
        self.t = 0.0

        return

    def updateCar(self,dt, throttle, steering, external_states=None): 
        if external_states is None:
            x,y,v,heading = self.states
        else:
            x,y,v,heading = external_states
        beta = np.arctan( np.tan(steering) * self.lr / (self.lf+self.lr))
        dXdt = v * np.cos( heading + beta )
        dYdt = v * np.sin( heading + beta )
        dvdt = throttle
        dheadingdt = v/self.lr*np.sin(beta)

        x += dt * dXdt
        y += dt * dYdt
        v += dt * dvdt
        heading += dt * dheadingdt
        self.t += dt

        self.states = np.array([x,y,v,heading])
        new_states = {'coord':(x,y), 'heading':heading, 'vf':v, 'omega':0}
        return new_states


