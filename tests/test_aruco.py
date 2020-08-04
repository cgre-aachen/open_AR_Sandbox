from sandbox.markers import ArucoMarkers
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