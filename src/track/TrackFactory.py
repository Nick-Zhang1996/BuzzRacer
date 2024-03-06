from common import *
from track.Track import Track
#from track.RCPTrack import RCPTrack
from track.RCPTrackQpSmooth import RCPTrackQpSmooth as RCPTrack
from track.EmptyTrack import EmptyTrack
from track.Skidpad import Skidpad
from track.OrcaTrack import OrcaTrack
from track.CurvilinearTrack import CurvilinearTrack
from track.NascarTrack import NascarTrack
from track.SineTrack import SineTrack
from track.TriangleTrack import TriangleTrack
from math import radians


class TrackFactory:
    @staticmethod
    def getMapping():
        # NOTE skidpad, empty, orca weren't fully tested
        mapping = {
                'saved':TrackFactory.prepareSavedTrack,
                'nascar_saved':TrackFactory.prepareSavedNascarTrack,
                'sine_saved':TrackFactory.prepareSavedSineTrack,
                'triangle_saved':TrackFactory.prepareSavedTriangleTrack,
                # RCP tracks
                'full':TrackFactory.prepareRcpTrack,
                'small':TrackFactory.prepareRcpTrackSmall,
                'easy':TrackFactory.prepareEasyTrack,
                'big':TrackFactory.prepareRcpTrackBig,
                # curvilinear tracks
                'nascar':TrackFactory.prepareNascarTrack,
                'sine':TrackFactory.prepareSineTrack,
                'triangle':TrackFactory.prepareTriangleTrack,
                # specialized tracks
                'skidpad':TrackFactory.prepareSkidpad,
                'empty':TrackFactory.prepareEmptyTrack,
                'orca':TrackFactory.prepareOrcaTrack}
        return mapping

    @staticmethod
    def availableTrackNames():
        return TrackFactory.getMapping().keys()

    @staticmethod
    def build(name=None,*,main=None,config=None,track=None):
        ''' create a track
            name: name of track, if absent, use config
            main,config: parameters to assign to track
            track: if present, use as track instance (useful for initializing subclass of Track like QpSmooth(), if not, create a new one.
        '''
        mapping = TrackFactory.getMapping()
        if name is None:
            if (config is None):
                print_error('either specify [name] or specify a [config] that contains a track configuration')
            else:
                name = config.firstChild.nodeValue
        if (name in mapping):
            return mapping[name](main,config,track=track)
        else:
            print_error(f"unknown track name, use one in {mapping.keys()}")
            return

    @staticmethod
    def prepareSavedTrack(main,config,track=None):
        if (track is None):
            track = Track(main=main,config=config)
        track = track.load(main=main,config=config)
        print(f'loaded track of type {type(track)}')
        return track


    @staticmethod
    def prepareEmptyTrack(main,config):
        return EmptyTrack()

    @staticmethod
    def prepareRcpTrack(main,config,track=None):
        # row, col
        track_size = (6,4)
        if (track is None):
            track = RCPTrack(main=main,config=config)
        # drivable surface width 0.563, square tile side length 0.6
        track.initTrack('uuurrullurrrdddddluulddl',track_size, scale=0.6)
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
        #track.initRaceline((3,3),'d',offset=adjustment)
        track.initRaceline((3,3),'d',offset=None)
        #track.start_pos = (0.6*3.5,0.6*1.75)
        #track.start_dir = radians(90)
        return track

    @staticmethod
    def prepareSkidpad(main,config,track=None):
        track = Skidpad(main=main,config=config)
        return track

    @staticmethod
    def prepareRcpTrackSmall(main,config,track=None):
        # current track setup in mk103, L shaped
        # width 0.563, length 0.6
        if (track is None):
            track = RCPTrack()
        track.initTrack('uuruurddddll',(5,3),scale=0.6)
        # add manual offset for each control points
        adjustment = [0,0,0,0,0,0,0,0,0,0,0,0]
        adjustment[4] = -0.5
        adjustment[8] = -0.5
        adjustment[9] = 0
        adjustment[10] = -0.5
        track.initRaceline((2,2),'d',offset=adjustment)
        return track

    @staticmethod
    def prepareRcpTrackBig(main,config,track=None):
        if (track is None):
            track = RCPTrack()
        track.initTrack('urruulluururrdrdddlddlll',(7,5),scale=0.6)
        track.initRaceline((2,0),'l')
        return track

    @staticmethod
    def prepareEasyTrack(main,config,track=None):
        if (track is None):
            track = RCPTrack()
        #track.initTrack('uuruluurrrddddldll',(6,4),scale=0.6)
        #track.initRaceline((3,3),'d')
        track.initTrack('uuuuurrrddddldll',(6,4),scale=0.6)
        track.initRaceline((3,3),'d')
        return track

    @staticmethod
    def prepareCircle(main,config,track=None):
        if (track is None):
            track = RCPTrack()
        track_size = (6,6)
        track.initTrack('uuuururrrdrdddldllll',track_size, scale=0.6)
        track.initRaceline((0,2),'u',offset=None)
        return track

    @staticmethod
    def prepareOrcaTrack(main,config,track=None):
        track = OrcaTrack(main,config)
        return track

    @staticmethod
    def prepareNascarTrack(main,config,track=None):
        track = NascarTrack(main,config)
        return track

    @staticmethod
    def prepareSavedNascarTrack(main,config,track=None):
        if (track is None):
            track = Track(main=main,config=config)
        track = track.load(filename='nascar.p',main=main,config=config)
        return track

    @staticmethod
    def prepareSineTrack(main,config,track=None):
        track = SineTrack(main,config)
        return track

    @staticmethod
    def prepareSavedSineTrack(main,config,track=None):
        if (track is None):
            track = Track(main=main,config=config)
        track = track.load(filename='sine.p',main=main,config=config)
        return track

    @staticmethod
    def prepareTriangleTrack(main,config,track=None):
        track = TriangleTrack(main,config)
        return track
    @staticmethod
    def prepareSavedTriangleTrack(main,config,track=None):
        if (track is None):
            track = Track(main=main,config=config)
        track = track.load(filename='triangle.p',main=main,config=config)
        return track
