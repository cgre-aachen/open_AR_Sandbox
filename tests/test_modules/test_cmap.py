from sandbox import _calibration_dir as calib_dir
from sandbox import _test_data as test_data
from sandbox.sensor import Sensor
from sandbox.projector import CmapModule
import numpy as np

file = np.load(test_data['topo'] + "DEM1.npz")
frame = file['arr_0']
extent = np.asarray([0, frame.shape[1], 0, frame.shape[0], frame.min(), frame.max()])

import matplotlib.pyplot as plt

def test_init():
    module = CmapModule(extent=extent)
    print(module)

def test_render_frame():
    module = CmapModule(extent=extent)
    fig, ax = plt.subplots()
    module.render_frame(frame, ax)
    fig.show()

def test_change_cmap():
    module = CmapModule(extent=extent)
    fig1, ax1 = plt.subplots()
    module.render_frame(frame, ax1)
    fig1.show()
    cmap = plt.cm.get_cmap('Accent_r')
    fig2, ax2 = plt.subplots()
    module.set_cmap(cmap)
    module.render_frame(frame, ax2)
    fig2.show()

def test_widgets():
    module = CmapModule(extent=extent)
    widget = module.widgets_plot()
    widget.show()

def test_change_array():
    fig, ax = plt.subplots()
    col = ax.pcolormesh(frame)
    fig.show()

    file = np.load(test_data['topo'] + "DEM2.npz")
    frame2 = file['arr_0']
    col.set_array(frame2)
    fig.show()

def test_change_cmap():
    fig, ax = plt.subplots()
    col = ax.pcolormesh(frame)
    fig.show()

    cmap = plt.cm.get_cmap("hot")
    col.set_cmap(cmap)
    fig.show()

def test_delete_image():
    module = CmapModule(extent=extent)
    fig, ax = plt.subplots()
    module.render_frame(frame, ax)
    fig.show()

    module.delete_image()
    fig.show()




