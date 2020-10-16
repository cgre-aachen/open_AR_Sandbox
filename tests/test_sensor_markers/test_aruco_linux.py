from sandbox.markers.aruco_linux import *
from sandbox.sensor import KinectV2


def test_registration():
    kinect = KinectV2()
    set_device(kinect)
    _ = registration()
    plot_all()

def test_distort():
    kinect = KinectV2()
    set_device_params(kinect)
    x, y = distort(100, 100)
    assert x == 99.17397057586149 and y == 99.43425492981704

def test_depth_to_color():
    kinect = KinectV2()
    set_device_params(kinect)
    rx, ry = depth_to_color(100, 100, 10000)
    assert rx == 505.6134745126878 and ry == 241.7240982502678

def test_depth_to_camera():
    kinect = KinectV2()
    set_device(kinect)
    undistorted_depth, registered_RGB, big_depth = registration()
    points = depth_to_camera(undistorted_depth)
    assert np.asarray(points).shape == (3, 424, 512)

def test_create_map():
    kinect = KinectV2()
    set_device(kinect)
    set_device_params(kinect)
    undistorted_depth, _, _ = registration()
    _ = depth_to_camera(undistorted_depth)
    df = create_CoordinateMap(kinect.get_frame())
    assert len(df) > 1000

def test_start_mapping():
    kinect = KinectV2()
    df = start_mapping(kinect)
    assert len(df) > 1000
    plot_all(df=df, index=18900)