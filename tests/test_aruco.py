from sandbox.sensor import Sensor
from sandbox.markers import ArucoMarkers
import numpy as np

def test_init_arucos():
    sensor = Sensor(name = "kinectv2")
    aruco = ArucoMarkers(calibration=sensor.extent)
    print(aruco)

