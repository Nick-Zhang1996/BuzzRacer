# demonstrate qpSmooth on different tracks
from qp_smooth import QpSmooth

qp = QpSmooth()
qp.save_dir = '/home/nick/Buzzracer/'

track_id = 3

if (track_id == 0):
    # full track
    # number of rows, number of cols
    track_size = (6, 4)
    # starting from bottom left corner
    qp.init_track('uuurrullurrrdddddluulddl', track_size, scale=0.6)
    # coordinate of the initial grid, and the entry direction to it
    qp.init_raceline((3, 3), 'd', offset=None)
    qp.img_filename = 'track0.png'
elif (track_id == 1):
    # track 1
    track_size = (4, 3)
    qp.init_track('urulurrdddll', track_size, scale=0.6)
    qp.init_raceline((0, 0), 'l', offset=None)
    qp.img_filename = 'track1.png'
elif (track_id == 2):
    track_size = (8, 5)
    qp.init_track('ururuuululurrdrddrddddllll', track_size, scale=0.6)
    qp.init_raceline((0, 0), 'l', offset=None)
    qp.img_filename = 'track2.png'
elif (track_id == 3):
    track_size = (9, 7)
    qp.init_track('uuurrddrruuluulluulurrrrrrdldldrdrdddldlllll',
                  track_size, scale=0.6)
    qp.init_raceline((0, 0), 'l', offset=None)
    qp.img_filename = 'track3.png'

qp.optimize_path(visualize=False, save_steps=False)
