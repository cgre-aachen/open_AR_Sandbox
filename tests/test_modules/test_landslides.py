from sandbox import _test_data as test_data
from sandbox.modules import LandslideSimulation
import matplotlib.pyplot as plt

import numpy as np
file = np.load(test_data['topo'] + "DEM1.npz")
frame = file['arr_0']
extent = [0, frame.shape[1], 0, frame.shape[0], frame.min(), frame.max()]


def test_init():
    module = LandslideSimulation(extent=extent)
    print(module)

def test_update():
    module = LandslideSimulation(extent=extent)
    fig, ax = plt.subplots()
    depth, ax, size, cmap, norm = module.update(frame, ax, extent)
    ax.imshow(depth, vmin=size[-2], vmax=size[-1], cmap=cmap, norm=norm, origin='lower')
    fig.show()

def test_load_simulation():
    module = LandslideSimulation(extent=extent)
    module.load_simulation_data_npz(test_data['landslide_simulation']+'Sim_Topo1_Rel13_results4sandbox.npz')
    assert module.velocity_flow is not None and module.height_flow is not None

def test_load_release_area():
    module = LandslideSimulation(extent=extent)
    module.Load_Area.loadTopo(test_data['landslide_topo']+'Topography_3.npz')
    assert module.Load_Area.file_id == '3'

    module.load_release_area(test_data['landslide_release'])
    assert module.release_options == ['ReleaseArea_3_1.npy', 'ReleaseArea_3_3.npy', 'ReleaseArea_3_2.npy']
    assert module.release_id_all == ['1', '3', '2']

def test_show_box_release():
    module = LandslideSimulation(extent=extent)
    module.Load_Area.loadTopo(test_data['landslide_topo'] + 'Topography_3.npz')
    module.load_release_area(test_data['landslide_release'])
    module.modify_to_box_coordinates(id = '1')
    fig, ax = plt.subplots()
    ax.imshow(frame, vmin=extent[-2], vmax=extent[-1], cmap='gist_earth_r',origin='lower')
    module.show_box_release(ax, module.release_area)
    assert np.allclose(np.asarray([[74., 72.], [74., 84.],[86., 84.],[86., 72.]]), module.release_area)
    fig.show()

def test_plot_landslide():
    module = LandslideSimulation(extent=extent)
    module.load_simulation_data_npz(test_data['landslide_simulation'] + 'Sim_Topo1_Rel13_results4sandbox.npz')
    module.flow_selector = "Velocity"
    module.frame_selector = 10
    fig, ax = plt.subplots()
    ax.imshow(frame, vmin=extent[-2], vmax=extent[-1], cmap='gist_earth_r', origin='lower')
    module.plot_landslide_frame(ax)
    fig.show()

    module.flow_selector = "Height"
    ax.cla()
    ax.imshow(frame, vmin=extent[-2], vmax=extent[-1], cmap='gist_earth_r', origin='lower')
    module.plot_landslide_frame(ax)
    fig.show()

def test_panel_plot():
    module = LandslideSimulation(extent=extent)
    module.load_simulation_data_npz(test_data['landslide_simulation'] + 'Sim_Topo1_Rel13_results4sandbox.npz')
    module.frame_selector = 10
    module.plot_frame_panel()
    module.plot_flow_frame.show()

def test_show_widgets():
    module = LandslideSimulation(extent=extent)
    module.load_simulation_data_npz(test_data['landslide_simulation'] + 'Sim_Topo1_Rel13_results4sandbox.npz')
    module.flow_selector = "Velocity"
    module.frame_selector = 10
    load_save = module.Load_Area.show_widgets()
    module._create_widgets()
    landslide = module.widgets_controller_simulation()
    landslide.show()

