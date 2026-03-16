# Buzzracer

This repository contains various resources for the Buzzracer platform developed by DCSL at Georgia Tech. Buzzracer is a 1/24 scale autonomous vehicle platform for education and research. It allows easy portation between simulation and hardware, and supports multiple vehicles to run simultaneously

![Buzzracer](docs/topology.png)

## File Structure

The file structure is organized as follows

```
/docs : documents
/outputs  : State logs, gifs, snapshots etc.
/assets : Non-code assets, images, offline-optimized raceline etc.
/configs : Config files for configuring experiments or simulations
/src/buzzracer  : Main source
/src/buzzracer/extensions : extensions that can be loaded when running experiments
/src/buzzracer/controllers: controllers
/src/buzzracer/cars : car platform related modules
/src/buzzracer/sysid : System identification, vehicle dynamics model, etc.
/src/buzzracer/track : Definitions for race track
/src/buzzracer/RL : Reinforcement Learning related modules
/src/utillities : utilities
```

## Getting Started

This section will walk you through of getting the codebase running on your local machine

### Clone Repository

First clone this repository

```
git clone https://github.gatech.edu/zzhang615/RC-VIP'
```

or

```
git clone git@github.gatech.edu:zzhang615/RC-VIP.git'
```

depending on which type of authentication you plan to use. If you're accessing through `github.com` instead of `github.gatech.edu`, your URL will be different.

Note that ssh does not work with `github.gatech.edu` unless you are using campus network or connected to the GT VPN.

### Install Dependencies

We use `uv` for virtual dependency management, run `uv sync` to install necessary dependencies
With `uv`, run scripts with `uv run scripts/run.py stanley`.
You may also install the library with `pip install -e .`, then run `python scripts/run.py stanley`

### Generate Raceline

Once you have installed all required packages, you need to run `uv run scripts/qp_smooth.py` to generate a raceline profile. 
You should see some colorful text bring printed and several visualizations of our racetrack with a raceline. 
Click the 'x' on the upper corners for each visualization to continue the program. When the program finishes, the last two lines should be:

```
testing loading
track and raceline loaded
```

This means the raceline profile have been saved correctly

### Verify Your Installation

To verify everything is working, launch a simulation with

```
uv run scripts/run.py stanley
```

This runs an experiment with a single car using the stanley controller.

You should see a simulation of a car running around on screen
![Buzzracer](docs/sample_sim.gif)


`run.py` is the primary entry point for all simulation and experiments, and `stanley` refers to config file`configs/stanley.xml`. 
The config file contains details of an experiment, for example,  whether to run the experiment in simulation or in real world, which race track to load, how many cars to generate, which controller each car uses, which extension to load etc. `run.py` loads this config file and prepares everything accordingly. 


You can run a different config file, for example, one with multiple vehicles

```
uv run scripts/run.py planner_stanley
```

![Buzzracer](docs/sample_sim_multi.gif)

When you're working on your project, you will likely create a new controller, extension, visualization etc.
In order to test your module, you'll create your own config file and place it under `configs/`.

### Extensions

Visualization, logging, laptimer, collision monitor etc. are implemented as extensions. They are located under `src/buzzracer/extensions/` and can be loaded at runtime if specified in config xml files. Check `src/buzzracer/extensions/Extension.py` for the standard format

### Next Steps

Now that you have the repository properly set up, it's time to read the sources files to get a better understanding of how everything works together. To get you started, try reading all relevent codes for the test experiment you ran. You can check the relevant config file for the modules it invoked, to give you some ideas, start with the following files:

```
configs/stanley.xml
run.py
src/buzzracer/extensions/extension.py
src/buzzracer/extensions/__init__.py
src/buzzracer/extensions/laptimer.py
src/buzzracer/extensions/visualization.py
src/buzzracer/extensions/simulator.py
src/buzzracer/extensions/simulator/dynamic_bicycle_model.py
src/buzzracer/controllers/car_controller.py
src/buzzracer/controllers/stanley_car_controller.py
```
