import numpy as np
from time import time

import build.particle_game

A = np.eye(3)
B = np.ones((3,2))
x = np.array([1,2,3]).reshape((-1,1))
u = np.array([7,8]).reshape((-1,1))

def f(x,u):
    return A @ x + B @ u

game = build.particle_game.ParticleGame(1,1,1,1, 0.1,0.1,0.1,0.1,0.1, A,A,A,A,B,A,A)
t0 = time()

t0 = time()
result = game.f(x,u)
print(time()-t0)
print(result.shape)

t0 = time()
result = f(x,u)
print(result.shape)
print(time()-t0)

a = np.random.random((4,3,2))

print(a[1])
print(game.three_dim([aa for aa in a]))


