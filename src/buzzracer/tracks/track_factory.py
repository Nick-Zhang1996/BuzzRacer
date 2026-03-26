from buzzracer.common import get_logger, set_config_attr
from buzzracer.tracks.track import TrackConfig
from buzzracer.tracks.curvilinear_track import CurvilinearTrack
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
    def build(name=None, *, config=None):
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
            return mapping[name](config)
        else:
            logger.error('unknown track name, use one in %s', mapping.keys())

    @staticmethod
    def prepare_saved_track(config_dom):
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
    def prepare_rcp_track(config_dom):
        # row, col
        config = RCPTrack.build_config('uuurrullurrrdddddluulddl', GridSize(6, 4))
        set_config_attr(config_dom, config)
        track = RCPTrack(config)
        rcp_raceline = track.build_raceline((3, 3), 'd', offset=None)
        r_vec, left, right = track.process_rcp_raceline(rcp_raceline)
        data = track.build_track(r_vec, left, right)
        track.rcp_raceline = rcp_raceline
        track.data = data
        return track

    @staticmethod
    def prepare_skidpad(config_dom):
        config = TrackConfig()
        set_config_attr(config_dom, config)
        track = Skidpad(config)
        return track

    @staticmethod
    def prepare_rcp_track_small(config_dom):
        config = RCPTrack.build_config('uuruurddddll', GridSize(5, 3))
        set_config_attr(config_dom, config)
        track = RCPTrack(config)
        # Add manual offset for each control points
        adjustment = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        adjustment[4] = -0.5
        adjustment[8] = -0.5
        adjustment[9] = 0
        adjustment[10] = -0.5
        rcp_raceline = track.build_raceline((2, 2), 'd', offset=adjustment)
        r_vec, left, right = track.process_rcp_raceline(rcp_raceline)
        data = track.build_track(r_vec, left, right)
        track.rcp_raceline = rcp_raceline
        track.data = data
        return track

    @staticmethod
    def prepare_rcp_track_big(config_dom):
        config = RCPTrack.build_config('urruulluururrdrdddlddlll', GridSize(7, 5))
        set_config_attr(config_dom, config)
        track = RCPTrack(config)
        rcp_raceline = track.build_raceline((2, 0), 'l')
        r_vec, left, right = track.process_rcp_raceline(rcp_raceline)
        data = track.build_track(r_vec, left, right)
        track.rcp_raceline = rcp_raceline
        track.data = data
        return track

    @staticmethod
    def prepare_easy_track(config_dom):
        config = RCPTrack.build_config('uuuuurrrddddldll', GridSize(6, 4))
        set_config_attr(config_dom, config)
        track = RCPTrack(config)
        rcp_raceline = track.build_raceline((3, 3), 'd')
        r_vec, left, right = track.process_rcp_raceline(rcp_raceline)
        data = track.build_track(r_vec, left, right)
        track.rcp_raceline = rcp_raceline
        track.data = data
        return track

    @staticmethod
    def prepare_circle(config_dom):
        config = RCPTrack.build_config('uuuururrrdrdddldllll', GridSize(6, 6), scale=0.6)
        set_config_attr(config_dom, config)
        track = RCPTrack(config)
        rcp_raceline = track.build_raceline((0, 2), 'u', offset=None)
        r_vec, left, right = track.process_rcp_raceline(rcp_raceline)
        data = track.build_track(r_vec, left, right)
        track.rcp_raceline = rcp_raceline
        track.data = data
        return track

    @staticmethod
    def prepare_nascar_track(config_dom):
        config = TrackConfig()
        set_config_attr(config_dom, config)
        track = NascarTrack(config)
        return track
