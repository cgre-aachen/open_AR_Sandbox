from sandbox.projector import Projector
from sandbox import _calibration_dir as calib_dir
from sandbox import _test_data as test_data
import matplotlib.pyplot as plt
import numpy as np

from tests import test_calib_sensor

file = np.load(test_data['topo'] + "DEM1.npz")
frame = file['arr_0']
extent = np.asarray([0, frame.shape[1], 0, frame.shape[0], frame.min(), frame.max()])


def test_init_projector():
    projector = Projector(use_panel=False)
    assert projector.panel is not None

def test_save_load_calibration_projector():
    projector = Projector(use_panel=False)
    file = calib_dir + 'test_projector_calibration.json'
    projector.save_json(file=file)
    # now to test if it loads correctly the saved one
    projector2 = Projector(calibprojector = file, use_panel=False)

def test_open_panel_browser():
    projector = Projector(use_panel=False)
    projector.start_server()

def test_trigger():
    projector = Projector(use_panel=True)
    projector.ax.plot([10, 20, 30], [20, 39, 48])
    projector.trigger()

def test_delete_ax_image():
    projector = Projector(use_panel=False)
    projector.ax.plot([10, 20, 30], [20, 39, 48])
    projector.clear_axes()

def test_delete_points():
    projector = Projector(use_panel=False)
    line1 = projector.ax.plot([10, 20, 30], [20, 39, 48])
    projector.trigger()
    del line1
    projector.trigger()


def test_change_betweeen_axes():
    """Not the way to go"""
    fig_supl, ax_supl = plt.subplots()
    ax_supl.set_axis_off()
    ax_supl.pcolormesh(frame)
    fig_supl.show()

    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.plot([10, 50, 100], [10, 50, 100], 'k.')
    fig.show()

    ax.add_child_axes(ax_supl)
    fig.show()

    print(ax.child_axes)

    del ax.child_axes[0]
    fig.show()


def test_change_betweeen_axes2():
    """Show that cannot be done this way"""
    fig_supl, ax_supl = plt.subplots()
    ax_supl.set_axis_off()

    fig, ax = plt.subplots()
    ax.set_axis_off()

    ax_supl.cla()
    # ax.add_child_axes(ax_supl)
    fig.show()

    ax.pcolormesh(frame)
    fig.show()

    ax_supl.plot([10, 50, 100], [10, 50, 100], 'k.')
    fig.show()


def test_erase_axes_figure():
    figure, ax = plt.subplots()
    ax.set_axis_off()
    col = ax.pcolormesh(frame)
    figure.show()
    line, = ax.plot([10, 50, 100], [10, 50, 100], 'k.')
    figure.show()
    line.remove()
    figure.show()
    col.remove()
    figure.show()
    ax.plot([10,20],[10,201])
    figure.show()

def test_widgets_calibration():
    projector = Projector(use_panel=True)
    widget = projector.calibrate_projector()
    widget.show()

    projector2 = Projector(calibprojector=calib_dir + 'my_projector_calibration.json', use_panel=True)
    widget2 = projector2.calibrate_projector()
    widget2.show()

