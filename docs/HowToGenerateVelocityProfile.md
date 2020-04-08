## How to Calculate Velocity profile and Laptime
This document explains how to generate velocity profile for a given trajectory and calculate theoretical lap time based on that.

### Theory
One simple model of vehicle traction is Traction Circle, which states that at any instant, the norm of maximum acceleration a vehicle can generate is a fixed value. Stated graphically, it means that if one is to plot the lateral acceleration on x axis, and longitudinal acceleration on y axis, the achievable values lie inside a circle. This circle is called Traction Circle, and the radius of which is maximum acceleration. In road cars, the radius is somewhat around 1g.

![Traction Circle](../traction-circle.jpg)

### Implementation
Given a fixed trajectory, there are two limiting factors that govern speed profile, one being traction available, the other being vehicle's accelerating and braking capabilities. It is intuitive that in order to maximize speed, the vehicle has to utilize all traction available. This gives rise to a three pass algorithm, originally designed by Dr Tsiotras.

The first pass calculates speed limited sorely by lateral acceleration, which results from curvature of the tracjectory. The speed profile after first pass is called `v1`

```
        # let raceline curve be r(u)
        # dr = r'(u), parameterized with xx/u
        dr = np.array(splev(xx,self.raceline,der=1))
        # ddr = r''(u)
        ddr = np.array(splev(xx,self.raceline,der=2))
        _norm = lambda x:np.linalg.norm(x,axis=0)
        # radius of curvature can be calculated as R = |y'|^3/sqrt(|y'|^2*|y''|^2-(y'*y'')^2)
        curvature = 1.0/(_norm(dr)**3/(_norm(dr)**2*_norm(ddr)**2 - np.sum(dr*ddr,axis=0)**2)**0.5)

        # first pass, based on lateral acceleration
        v1 = (mu*g/curvature)**0.5

```

Plotted below is the acceleration throughout the trajectory, if the vehicle were to follow `v1`

![First Pass](../v1.png)

The second pass takes into account vehicle's ability to accelerate, starting from the slowest point in `v1`, it iterates forward and constrains vehicle's velocity to available traction and engine's acceleration capabilities.

```
        dist = lambda a,b: ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5
        # second pass, based on engine capacity and available longitudinal traction
        # start from the index with lowest speed
        min_xx = np.argmin(v1)
        v2 = np.zeros_like(v1)
        v2[min_xx] = v1[min_xx]
        for i in range(min_xx,min_xx+n_steps):
            # lateral acc at next step if the car mainains speed
            a_lat = v2[i%n_steps]**2*curvature[(i+1)%n_steps]

            # is there available traction for acceleration?
            if ((mu*g)**2-a_lat**2)>0:
                a_lon_available_traction = ((mu*g)**2-a_lat**2)**0.5
                # constrain with motor capacity
                a_lon = min(acc_max_motor(v2[i%n_steps]),a_lon_available_traction)

                (x_i, y_i) = splev(xx[i%n_steps], self.raceline, der=0)
                (x_i_1, y_i_1) = splev(xx[(i+1)%n_steps], self.raceline, der=0)
                # distance between two steps
                ds = dist((x_i, y_i),(x_i_1, y_i_1))
                # assume vehicle accelerate uniformly between the two steps
                v2[(i+1)%n_steps] =  min((v2[i%n_steps]**2 + 2*a_lon*ds)**0.5,v1[(i+1)%n_steps])
            else:
                v2[(i+1)%n_steps] =  v1[(i+1)%n_steps]

        v2[-1]=v2[0]
```

Plotted below is the acceleration throughout the trajectory, if the vehicle were to follow `v2`

![Second Pass](../v2.png)

The third pass takes into account vehicle's ability to decelerate, starting from the slowest point in `v2`, it iterates backward and constrains vehicle's velocity to available traction and vehicle's braking capabilities.

```
        # third pass, backwards for braking
        min_xx = np.argmin(v2)
        v3 = np.zeros_like(v1)
        v3[min_xx] = v2[min_xx]
        for i in np.linspace(min_xx,min_xx-n_steps,n_steps+2):
            i = int(i)
            a_lat = v3[i%n_steps]**2*curvature[(i-1+n_steps)%n_steps]
            a_lon_available_traction = abs((mu*g)**2-a_lat**2)**0.5
            a_lon = min(dec_max_motor(v3[i%n_steps]),a_lon_available_traction)
            #print(a_lon)

            (x_i, y_i) = splev(xx[i%n_steps], self.raceline, der=0)
            (x_i_1, y_i_1) = splev(xx[(i-1+n_steps)%n_steps], self.raceline, der=0)
            # distance between two steps
            ds = dist((x_i, y_i),(x_i_1, y_i_1))
            #print(ds)
            v3[(i-1+n_steps)%n_steps] =  min((v3[i%n_steps]**2 + 2*a_lon*ds)**0.5,v2[(i-1+n_steps)%n_steps])
            #print(v3[(i-1+n_steps)%n_steps],v2[(i-1+n_steps)%n_steps])
            pass

        v3[-1]=v3[0]
```

Plotted below is the acceleration throughout the trajectory, if the vehicle were to follow `v3`

![Third Pass](../v3.png)

This is the final velocity profile, a cubic spline is fitted to smooth out the profile, since the original profile is calculated discretely.

```
        self.v3 = interp1d(xx,v3,kind='cubic')
```

It is then possible to calculate laptime for a vehicle following `v3`


```
        # calculate theoretical lap time
        t_total = 0
        for i in range(n_steps):
            (x_i, y_i) = splev(xx[i%n_steps], self.raceline, der=0)
            (x_i_1, y_i_1) = splev(xx[(i+1)%n_steps], self.raceline, der=0)
            # distance between two steps
            ds = dist((x_i, y_i),(x_i_1, y_i_1))
            t_total += ds/v3[i%n_steps]
        print("top speed = %.2fm/s"%max(v3))
        print("total time = %.2fs"%t_total)

```

These are implemented in `track.py` in function `initRaceline()`, make sure you use the snippet from master branch
