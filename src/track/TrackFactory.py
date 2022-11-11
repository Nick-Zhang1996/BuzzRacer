from common import *
from track.Track import Track
from track.RCPTrack import RCPTrack
from track.EmptyTrack import EmptyTrack
from track.Skidpad import Skidpad
from track.OrcaTrack import OrcaTrack
from math import radians
def TrackFactory(main,config,name=None):
    if name is None:
        name = config.firstChild.nodeValue
    mapping = {'full':prepareRcpTrack, 'small':prepareRcpTrackSmall, 'skidpad':prepareSkidpad, 'empty':prepareEmptyTrack,'orca':prepareOrcaTrack}
    if (name in mapping):
        return mapping[name](main,config)
    else:
        print_error(f"unknown track name, use one in {mapping.keys()}")
        return

def prepareEmptyTrack(main,config):
    return EmptyTrack()

def prepareRcpTrack(main,config):

    # width 0.563, square tile side length 0.6

    # full RCP track
    # NOTE load track instead of re-constructing
    fulltrack = RCPTrack(main=main,config=config)
    fulltrack.startPos = (0.6*3.5,0.6*1.75)
    fulltrack.startDir = radians(90)
    fulltrack.load()
    '''
    for key,value in config.attributes.items():
        setattr(fulltrack,key,eval(value))
    '''
    return fulltrack

    # row, col
    track_size = (6,4)
    #fulltrack.initTrack('uuurrullurrrdddddluulddl',track_size, scale=0.565)
    fulltrack.initTrack('uuurrullurrrdddddluulddl',track_size, scale=0.6)
    # add manual offset for each control points
    adjustment = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    adjustment[0] = -0.2
    adjustment[1] = -0.2
    #bottom right turn
    adjustment[2] = -0.2
    adjustment[3] = 0.5
    adjustment[4] = -0.2

    #bottom middle turn
    adjustment[6] = -0.2

    #bottom left turn
    adjustment[9] = -0.2

    # left L turn
    adjustment[12] = 0.5
    adjustment[13] = 0.5

    adjustment[15] = -0.5
    adjustment[16] = 0.5
    adjustment[18] = 0.5

    adjustment[21] = 0.35
    adjustment[22] = 0.35

    # start coord, direction, sequence number of origin
    # pick a grid as the starting grid, this doesn't matter much, however a starting grid in the middle of a long straight helps
    # to find sequence number of origin, start from the start coord(seq no = 0), and follow the track, each time you encounter a new grid it's seq no is 1+previous seq no. If origin is one step away in the forward direction from start coord, it has seq no = 1
    fulltrack.initRaceline((3,3),'d',10,offset=adjustment)
    return fulltrack

def prepareSkidpad(main,config):
    target_velocity = 1.0
    sp = Skidpad()
    sp.initSkidpad(radius=2,velocity=target_velocity)
    return sp

def prepareRcpTrackSmall(main,config):
    # current track setup in mk103, L shaped
    # width 0.563, length 0.6
    mk103 = RCPTrack()
    mk103.initTrack('uuruurddddll',(5,3),scale=0.57)
    # add manual offset for each control points
    adjustment = [0,0,0,0,0,0,0,0,0,0,0,0]
    adjustment[4] = -0.5
    adjustment[8] = -0.5
    adjustment[9] = 0
    adjustment[10] = -0.5
    mk103.initRaceline((2,2),'d',4,offset=adjustment)
    return mk103

def prepareOrcaTrack(main,config):
    track = OrcaTrack(main,config)
    return track
