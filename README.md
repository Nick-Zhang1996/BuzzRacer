# RC-VIP project
# VIP RC project

Welcome to the RC subteam repo. This is part of Georgia Tech's VIP team -- Autonomous and Semi-autonomous Vehicle Safety, led by professor P. Tsiotras

This readme is updated Aug 31 2020 to reflect the latest change to the code base. The repo contains some files that are not actively being used due to historical reasons, please ignore the files that are not mentioned in this readme for now.

## Overview

The platform you will be working on are the off-board platforms which are only passive actuators of throttle and steering commands. We have hardware that can plug into your computer's USB socket, and send command to the cars. You will have access to the vehicle's state, like vehicle's position, orientation, and speed. This is accomplished through our optitrack visual tracking system and an EKF (still in progress)

In an actual experiment that involves real cars driving on real track, the optitrack system will capture vehicles' state, relay them to your computer, and you computer would process them, and send appropriate control command via our special hardware to the cars. The cars would move, and the optitrack would find new states, and the process repeats.

Now, since nobody likes to come to lab to spend hours finding silly bugs, it is important to be able to run the pipeline locally, at the comfort of your home, without having to connect to all the equipment at lab. We've created a simple simulator that allows you verify that your code works, at least grammarly and kinematically. To see how this works in action, run `python src/run.py` after you install all the dependencies.

Now, to facilitate easy switching between running with the simulator and with actual hardware, we want our code to be modular. So the switching can be as simple as changing a single line of code, from using the simulator module to the realExperiment module. Something like that.

Being modular also allow us to develop an alternative for a particular part of the code, and quickly integrate into the pipeline. This can be the state streamer, the trajectory server, the control algorithm, the visualization, etc. for example students could work on their own control algorithm without worrying about the rest of pipeline

## Dependencies

This project is build mainly written in Python3 and C/C++, with C/C++ on PCB firmware only.

I've long lost track of the list of packages we use, or the steps used to install them. Most should come with pip. Google is your friend.

To name a few packages, python-opencv, pickle, numpy, scipy, PIL...

Run `src/run.py` and see what you're missing. 

## Important Files
`src/run.py' The entry point for running experiments, either for real or in simulation. If you run it as is it should show you a racetrack, with a car following trajectory. The black line denotes vehicle's heading, and the red line denotes the current steering angle.

`src/Track.py` Template for a Track object

`src/Skidpad.py` A subclass of Track, a skidpad is a circular track. It is used frequently in automotive testing to study handling characteristics for a car.

`src/RCPTrack.py` A subclass of Track. RCP track is a type of track that consists of multiple tiles. Each tile can be a straight or 90 deg bend section. They can be assembled into various combination tracks. This is the one that will be most frequently used as it is the track we have in our lab. It also has member functions to generate a raceline/trajectory, and provide information regarding a trajectory to facilitate control. In addition it contains some visualization implementations.

`src/car.py` Interface with cars and where ctrlCar(), the control algorithm lives.

Above are all you need to worry about right now

## Getting Started

Run `src/run.py` to verify you have installed all the packages, then read the code thoroughly and focus on what each class/function does. If you are asked to develop a new control algorithm, you will likely implement your own versions of `car.ctrlCar()` and `track.localTrajectory()`. The current versions are implemented loosely according to the Stanly's control algorithm, it's not a big deal if you don't get it.

If you run into problems, shoot me an email at (nickzhang at GA Tech dot edu)

Have fun and good luck!
