from sandbox import _test_data as test_data
from sandbox.sensor import CalibSensor

import numpy as np
file = np.load(test_data['topo'] + "DEM1.npz")
frame = file['arr_0']
extent = np.asarray([0, frame.shape[1], 0, frame.shape[0], frame.min(), frame.max()])

import matplotlib.pyplot as plt


def test_calibration():
    calib = CalibSensor(name = "dummy")

def test_create_widgets():
    calib = CalibSensor(name = "dummy", use_panel=False)
    calib._create_widgets()

def test_widgets():
    calib = CalibSensor(name="dummy", use_panel=False)
    widget = calib.calibrate_sensor()
    widget.show()
