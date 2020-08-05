from sandbox.markers import ArucoMarkers
from sandbox.sensor import Sensor
from sandbox import _test_data
im_folder = _test_data['test']
import numpy as np
import matplotlib.pyplot as plt
frame = np.load(im_folder+'frame1.npz')
depth = frame['arr_0']
color = frame['arr_1']

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

def test_plot_aruco_location_rgb():
    aruco = ArucoMarkers(sensor=Sensor(name="kinect_v2"))