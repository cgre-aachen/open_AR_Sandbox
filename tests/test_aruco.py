from sandbox.markers import ArucoMarkers, MarkerDetection
from sandbox.sensor import Sensor
from sandbox import _test_data, _calibration_dir
im_folder = _test_data['test']
import numpy as np
import matplotlib.pyplot as plt
frame = np.load(im_folder+'frame1.npz')
depth = frame['arr_0']
color = frame['arr_1']
try:
    sensor = Sensor(calibsensor= _calibration_dir + "sensorcalib.json", name='kinect_v2')
except:
    import warnings as warn
    warn("Testing will be performed without the sensor")
    sensor = None


def test_plot_image():
    depth = frame['arr_0']
    col = frame['arr_1']
    plt.imshow(depth)
    plt.show()
    plt.imshow(col)
    plt.show()

def test_aruco_detect():
    aruco = ArucoMarkers()
    corners, ids, rejected = aruco.aruco_detect(color)
    print(corners, ids, rejected)

def test_get_location_marker():
    aruco = ArucoMarkers()
    corners, _, _ = aruco.aruco_detect(color)
    x, y = aruco.get_location_marker(corners[0])
    print(x,y)

def test_find_markers_rgb():
    aruco = ArucoMarkers()
    rgb_markers = aruco.find_markers_rgb(color=color, amount=2)
    print(rgb_markers)

def test_find_markers_projector():
    aruco = ArucoMarkers()
    projector_markers, corner_middle = aruco.find_markers_projector(color=color, amount=2)
    print(corner_middle, projector_markers)

def test_create_aruco_marker():
    aruco = ArucoMarkers()
    aruco_array= aruco.create_aruco_marker(id=12, resolution=100, show=True, save=True, path=im_folder+'temp/')
    print(aruco_array)

def test_create_several_aruco_pdf():
    aruco = ArucoMarkers()
    fig = aruco.create_arucos_pdf(nx=5, ny=5, resolution=150, path=im_folder+'temp/')
    fig.show()

def test_update_arucos():
    aruco = ArucoMarkers(sensor=sensor)
    markers_in_frame = aruco.search_aruco(color)
    print(markers_in_frame)
    aruco.update_marker_dict()
    print(aruco.aruco_markers)
    aruco.transform_to_box_coordinates()
    print(aruco.aruco_markers)
    assert len(aruco.aruco_markers) == 2

def test_find_markers_IR():
    """Carefull. Most of the times doesn't work cause is stuck in infinite loop. If want to try, change the None with the number of arucos to detect"""
    aruco = ArucoMarkers(sensor=sensor)
    ir = aruco.find_markers_ir(amount=None)
    print(ir)

def test_create_coordinate_map():
    aruco = ArucoMarkers(sensor=sensor)
    map = aruco.create_CoordinateMap()
    print(map)

def test_marker_detection_class():
    """Work only with sensor active"""
    markers = MarkerDetection(sensor=sensor)
    fig, ax = plt.subplots()
    df = markers.update(frame=color)
    markers.plot_aruco(ax, df)
    fig.show()

def test_save_df():
    markers = MarkerDetection(sensor=sensor)
    df = markers.update(frame=color)
    df.to_pickle(im_folder+"temp/arucos.pkl")
    print(df)

def test_load_df():
    import pandas as pd
    df = pd.read_pickle(im_folder+"temp/arucos.pkl")
    print(df)

def test_widgets():
    markers = MarkerDetection(sensor=sensor)
    fig, ax = plt.subplots()
    df=markers.update(frame=color)
    markers.plot_aruco(ax, df)
    fig.show()
    widget = markers.widgets_aruco()
    widget.show()



