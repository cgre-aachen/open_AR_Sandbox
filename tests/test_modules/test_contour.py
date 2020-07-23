from sandbox import _test_data as test_data

from sandbox.projector import ContourLinesModule

import numpy as np
file = np.load(test_data['topo'] + "DEM1.npz")
frame = file['arr_0']
extent = np.asarray([0, frame.shape[1], 0, frame.shape[0], frame.min(), frame.max()])

import matplotlib.pyplot as plt


def test_init():
    module = ContourLinesModule(extent=extent)
    print(module)


def test_generating_all():
    module = ContourLinesModule(extent=extent)

    fig, ax = plt.subplots()
    module.plot_contour_lines(frame, ax)
    fig.show()

def test_create_widgets_plot():
    module = ContourLinesModule(extent=extent)
    widget = module.widgets_plot()
    widget.show()

def test_update_array():
    module = ContourLinesModule(extent=extent)
    fig, ax = plt.subplots()
    module.plot_contour_lines(frame, ax)
    fig.show()

    file = np.load(test_data['topo'] + "DEM2.npz")
    frame2 = file['arr_0']
    module.set_array(frame2)
    fig.show()

def test_delete_contours():
    module = ContourLinesModule(extent=extent)
    fig, ax = plt.subplots()
    module.plot_contour_lines(frame, ax)
    fig.show()
    module.delete_contourns(ax)
    fig.show()


