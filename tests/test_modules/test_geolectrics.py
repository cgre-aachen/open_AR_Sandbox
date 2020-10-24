from sandbox import _test_data
from sandbox.modules import GeoelectricsModule
import matplotlib.pyplot as plt
import pytest
import numpy as np
file = np.load(_test_data['topo'] + "DEM2.npz")
frame = (file['arr_0'] - file['arr_0'].min()) *1000+50
extent = [0, frame.shape[1], 0, frame.shape[0], frame.min(), frame.max()]

fig, ax = plt.subplots()
pytest.sb_params = {'frame': frame,
                    'ax': ax,
                    'fig': fig,
                    'extent': extent,
                    'marker': [],
                    'cmap': plt.cm.get_cmap('viridis'),
                    'norm': None,
                    'active_cmap': True,
                    'active_contours': True}

def test_init():
    geo = GeoelectricsModule()

def test_create_mesh():
    geo = GeoelectricsModule()
    _ = geo.create_mesh(pytest.sb_params["frame"], step = 7.5)
    geo.show_mesh()

def test_create_data_containerERT():
    geo = GeoelectricsModule()
    _=geo.create_mesh(pytest.sb_params["frame"], step=7.5)
    markers = np.array(([20, 130],
                 [160, 20],
                 [50, 60],
                 [150, 120]))
    geo.set_electrode_positions(markers)
    measurements = np.array(([0, 1, 2, 3],))
    scheme_type = "abmn"
    _=geo.create_data_containerERT(measurements, scheme_type)
    print(geo.scheme)

def test_calculate_current_flow():
    geo = GeoelectricsModule()
    _=geo.create_mesh(pytest.sb_params["frame"], step=7.5)
    markers = np.array(([20, 130],
                        [160, 20],
                        [50, 60],
                        [150, 120]))
    geo.set_electrode_positions(markers)
    measurements = np.array(([0, 1, 2, 3],))
    scheme_type = "abmn"
    _=geo.create_data_containerERT(measurements, scheme_type)
    _=geo.calculate_current_flow()
    geo.show_streams()

def test_calculate_resistivity():
    geo = GeoelectricsModule()
    _=geo.create_mesh(pytest.sb_params["frame"], step=7.5)
    markers = np.array(([20, 130],
                        [160, 20],
                        [50, 60],
                        [150, 120]))
    geo.set_electrode_positions(markers)
    measurements = np.array(([0, 1, 2, 3],))
    scheme_type = "abmn"
    _=geo.create_data_containerERT(measurements, scheme_type)
    _=geo.calculate_current_flow()
    _=geo.calculate_sensitivity()
    geo.show_sensitivity()

def test_update_resistivity():
    geo = GeoelectricsModule()
    geo.set_electrode_positions()
    geo.update_resistivity(pytest.sb_params["frame"],
                           pytest.sb_params["extent"],
                           7.5)
    geo.calculate_sensitivity()
    geo.show_streams()
    geo.show_sensitivity()

def test_plot_sandbox():
    geo = GeoelectricsModule()
    geo.set_electrode_positions()
    geo.sensitivity = True
    frame_r = pytest.sb_params["frame"]
    geo.update_resistivity(frame_r,
                           pytest.sb_params["extent"],
                           7.5)
    geo.calculate_sensitivity()

    fig, ax = plt.subplots()
    geo.stream = True
    geo.view = "mesh"
    ax = geo.plot(ax, frame_r, pytest.sb_params["cmap"])
    fig.show()

    geo.view = "potential"
    ax = geo.plot(ax, frame_r, pytest.sb_params["cmap"])
    fig.show()

    geo.stream = False
    geo.view = "sensitivity"
    ax = geo.plot(ax, frame_r, pytest.sb_params["cmap"])
    fig.show()
