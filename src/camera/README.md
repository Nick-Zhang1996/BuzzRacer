### Video Enhancement
this folder contains code for enhancing experiment videos. In particular, it allows dynamic video annotations using state logs recorded alongside the video. It's possible to correlate vehicle's physical position from the state log with vehicle's image position in the video. 

### Usage

First, state logs in format of a pickle file, and experiment video in a format readable by opencv (platform dependent) needs to be placed under `resources/` folder

Then, four objecet points (points in physical space) and their corresponding image points (pixel coordinate in video frame) needs to be recorded and put in transformation.py. 


Then, user needs to modify video.py to include the correct log file and video file name, and also specify the correct time offset, that is, video time where the experiment begins. After the script is run, user should see an annotated video. The video will be saved at outputs/, the folder need to be created beforehand
