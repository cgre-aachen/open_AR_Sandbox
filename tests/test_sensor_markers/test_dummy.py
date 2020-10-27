import numpy as np
from sandbox.markers import ArucoMarkers, MarkerDetection
from sandbox.markers.dummy_aruco import dummy_markers_in_frame
from sandbox import _test_data
file = np.load(_test_data['topo'] + "DEM1.npz")
frame = file['arr_0']


def test_detect_marker():
    dict_position = {1:[10,20],
          20:[30,20],
          2:[100,100],
          10:[40,50],
          5:[200,200],
          19:[1000,1000],
          99:[-10, 10]}

    df = dummy_markers_in_frame(dict_position, frame)
    print(df)
    assert df.loc[19, "is_inside_box"] == False

    xy = {}
    frm = np.empty((2,2))

    df = dummy_markers_in_frame(xy, frm)
    print(df)

def test_aruco_class():
    aruco = ArucoMarkers(sensor="dummy")
    assert aruco.kinect =="dummy"
    aruco.search_aruco(dict_position = {1:[10,20],2:[100,200]}, depth_frame=frame)
    print(aruco.markers_in_frame)

def test_marker_detection_class():
    aruco = MarkerDetection(sensor="dummy")
    df = aruco.update()
    print(df)
    aruco.set_aruco_position(dict_position = {1:[10,20],2:[100,50]}, frame=frame)
    df = aruco.update()
    print(df)
    aruco.set_aruco_position(dict_position={2:[100, 50], 4:[1000,1000]}, frame=frame)
    df = aruco.update()
    print(df)
    for i in range(10):
        df = aruco.update()
    print(df)



