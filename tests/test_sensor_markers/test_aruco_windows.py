from sandbox.markers.aruco_windows import *
from sandbox.sensor import KinectV2

def test_create_map():
    kinect = KinectV2()
    set_device(kinect)
    df = create_CoordinateMap(kinect.get_frame())
    assert len(df) > 1000

def test_start_mapping():
    kinect = KinectV2()
    df = start_mapping(kinect)
    assert len(df) > 1000