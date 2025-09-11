''' Test qp smooth '''
from buzzracer.scripts.qp_smooth import QpSmooth
from buzzracer.tracks.track_factory import TrackFactory


def test_qp_smooth():
    # optimize and save
    track: QpSmooth = QpSmooth()
    track = TrackFactory.build('full', track=track)
    track.optimize_path(max_iter=2, offset=0.1)
