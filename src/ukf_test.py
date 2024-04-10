from util.ukf import UKF
import numpy as np

if __name__=='__main__':
    ukf = UKF(init_val=2)
    truth = 0.8
    def h(x):
        return np.array([x**3+x**2, 0, 8, (2*x+5)*np.exp(x)])

    for i in range(10):
        sigma = ukf.getSigmaPoints()
        zs = np.array([h(s) for s in sigma]).T
        zx = h(truth).reshape((-1,1))
        ukf.update(zs,zx)
        print(ukf.mean,ukf.cov)



