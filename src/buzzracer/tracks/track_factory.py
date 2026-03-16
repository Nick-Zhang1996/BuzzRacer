from buzzracer.common import get_logger, set_config_attr
from buzzracer.tracks.track import TrackConfig
from buzzracer.tracks.rcp_track import RCPTrack, GridSize
from buzzracer.tracks.empty_track import EmptyTrack
from buzzracer.tracks.skidpad import Skidpad
from buzzracer.tracks.nascar_track import NascarTrack

logger = get_logger('TrackFactory')


class TrackFactory:
    @staticmethod
    def get_mapping():
        # NOTE skidpad, empty, orca weren't fully tested
        mapping = {
            # RCP tracks
            'saved': TrackFactory.prepare_saved_track,
            'full': TrackFactory.prepare_rcp_track,
            'small': TrackFactory.prepare_rcp_track_small,
            'easy': TrackFactory.prepare_easy_track,
            'big': TrackFactory.prepare_rcp_track_big,
            # specialized tracks
            'skidpad': TrackFactory.prepare_skidpad,
            'empty': TrackFactory.prepare_empty_track,
            'nascar': TrackFactory.prepare_nascar_track}
        return mapping

    @staticmethod
    def available_track_names():
        return TrackFactory.get_mapping().keys()

    @staticmethod
    def build(name=None, *, main=None, config=None, track=None):
        ''' create a track
            name: name of track, if absent, use config
            main,config: parameters to assign to track
            track: if present, use as track instance (useful for initializing subclass of Track like QpSmooth(), if not, create a new one.
        '''
        mapping = TrackFactory.get_mapping()
        if name is None:
            if (config is None):
                logger.error(
                    'either specify [name] or specify a [config] that contains a track configuration')
            else:
                name = config.firstChild.nodeValue
        if (name in mapping):
            return mapping[name](config, track=track)
        else:
            logger.error('unknown track name, use one in %s', mapping.keys())

    @staticmethod
    def prepare_saved_track(config_dom, track=None):
        if (track is None):
            config = TrackConfig()
            set_config_attr(config_dom, config)
            track = RCPTrack(config)
        track.load()
        return track

    @staticmethod
    def prepare_empty_track(config_dom):
        config = TrackConfig()
        set_config_attr(config_dom, config)
        return EmptyTrack(config)

    @staticmethod
    def prepare_rcp_track(config_dom, track=None):
        # row, col
        track_size = GridSize(6, 4)
        if (track is None):
            config = TrackConfig()
            set_config_attr(config_dom, config)
            track = RCPTrack(config)
        # drivable surface width 0.563, square tile side length 0.6
        track.init_track('uuurrullurrrdddddluulddl', track_size, scale=0.6)
        # add manual offset for each control points
        adjustment = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        adjustment[0] = -0.2
        adjustment[1] = -0.2
        # bottom right turn
        adjustment[2] = -0.2
        adjustment[3] = 0.5
        adjustment[4] = -0.2

        # bottom middle turn
        adjustment[6] = -0.2

        # bottom left turn
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
        # track.init_raceline((3,3),'d',offset=adjustment)
        track.init_raceline((3, 3), 'd', offset=None)
        # track.start_pos = (0.6*3.5,0.6*1.75)
        # track.start_dir = radians(90)
        return track

    @staticmethod
    def prepare_skidpad(config_dom, track=None):
        config = TrackConfig()
        set_config_attr(config_dom, config)
        track = Skidpad(config)
        return track

    @staticmethod
    def prepare_rcp_track_small(config_dom, track=None):
        # current track setup in mk103, L shaped
        # width 0.563, length 0.6
        if (track is None):
            config = TrackConfig()
            set_config_attr(config_dom, config)
            track = RCPTrack(config)
        track.init_track('uuruurddddll', GridSize(5, 3), scale=0.6)
        # add manual offset for each control points
        adjustment = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        adjustment[4] = -0.5
        adjustment[8] = -0.5
        adjustment[9] = 0
        adjustment[10] = -0.5
        track.init_raceline((2, 2), 'd', offset=adjustment)
        return track

    @staticmethod
    def prepare_rcp_track_big(config_dom, track=None):
        if (track is None):
            config = TrackConfig()
            set_config_attr(config_dom, config)
            track = RCPTrack(config)
        track.init_track('urruulluururrdrdddlddlll', GridSize(7, 5), scale=0.6)
        track.init_raceline((2, 0), 'l')
        return track

    @staticmethod
    def prepare_easy_track(config_dom, track=None):
        if (track is None):
            config = TrackConfig()
            set_config_attr(config_dom, config)
            track = RCPTrack(config)
        # track.init_track('uuruluurrrddddldll',GridSize(6,4),scale=0.6)
        # track.init_raceline((3,3),'d')
        track.init_track('uuuuurrrddddldll', GridSize(6, 4), scale=0.6)
        track.init_raceline((3, 3), 'd')
        return track

    @staticmethod
    def prepare_circle(config_dom, track=None):
        if (track is None):
            config = TrackConfig()
            set_config_attr(config_dom, config)
            track = RCPTrack(config)
        track_size = GridSize(6, 6)
        track.init_track('uuuururrrdrdddldllll', track_size, scale=0.6)
        track.init_raceline((0, 2), 'u', offset=None)
        return track

    @staticmethod
    def prepare_nascar_track(config_dom, track=None):
        config = TrackConfig()
        set_config_attr(config_dom, config)
        track = NascarTrack(config)
        return track
