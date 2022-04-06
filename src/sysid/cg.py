import numpy as np
# calculate CG location from measurement

# weight on each side, unit:grams
Gf = 70.26
Gr = 96.78
G = Gf + Gr
Gl = 82.67
Gr = 82.28
# wheelbase, mm
L = wb = 90
# track width, mm
w = tw = 59
tire_w = 10

# CG longitudinal location
lf = np.mean( [ wb*Gr/G, wb*(1-Gf/G) ] )
lr = wb - lf
print("lf = %.2f, lr = %.2f"%(lf,lr))

# CG Height, measured by raising one end of car

# doesn't really work...
'''
# amount of rise in front axle
delta = [8,-8,8,-8,12.7]
# which axle is measured, front:1, rear:-1
meas = [-1,1,1,-1,-1]
Gs = [97.31, 70.90, 69.56, 96.08, 90.52]
hs = []
for i in range(len(delta)):
    theta = np.arcsin(float(delta[i])/wb)
    if meas[i]>0:
        h = -(Gs[i]/G * wb - lr) / np.tan(theta)
    else:
        h = (Gs[i]/G * wb - lf) / np.tan(theta)
    hs.append(h)
print(hs)
'''

ll = Gr / (Gl+Gr) * tw
# amount of rise in left axle
delta = [-13,13,34.5,-34.5]
# which axle is measured, lef:1, right:-1
meas = [1,-1,-1,1]
Gs = [87.65,84.61,97.23,99.4]
hs = []
for i in range(len(delta)):
    theta = np.arcsin(float(-delta[i])/tw)
    if (meas[i]>0):
        Gr = G - Gs[i]
    else:
        Gr = Gs[i]
    h = ( (ll+ tire_w/2) - Gr/G*tw )/np.tan(theta)
    hs.append(h)
print(hs)

