import matplotlib.pyplot as plt
import numpy as np
# motor/longitudinal model
Cm1 = 6.03154
Cm2 = 0.96769
Cr = -0.20375
Cd = 0.00000

vv = np.linspace(0,10)
dvv = (Cm1-Cm2*vv)*0.7 - Cr - Cd*vv*vv
plt.plot(vv,dvv)
plt.show()
