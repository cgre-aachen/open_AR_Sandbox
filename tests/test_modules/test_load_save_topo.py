from sandbox import _test_data as test_data
from sandbox.modules import LoadSaveTopoModule
import matplotlib.pyplot as plt

import numpy as np
file = np.load(test_data['topo'] + "DEM1.npz")
frame = file['arr_0']
extent = [0, frame.shape[1], 0, frame.shape[0], frame.min(), frame.max()]


def test_init():
    module = LoadSaveTopoModule(extent=extent)
    print(module)

def test_update():
    module = LoadSaveTopoModule(extent=extent)
    fig, ax = plt.subplots()
    module.box_width = 100
    module.box_height = 50
    depth, ax, size, cmap, norm = module.update(frame, ax, extent)

    ax.imshow(depth, vmin=size[-2], vmax=size[-1], cmap="gist_earth_r", norm=norm, origin='lower')
    fig.show()

def test_plot_release_area():
    module = LoadSaveTopoModule(extent=extent)
    fig, ax = plt.subplots()
    module.box_width = 100
    module.box_height = 50
    module.add_release_area_origin(x=20, y=40)
    depth, ax, size, cmap, norm = module.update(frame, ax, extent)

    ax.imshow(depth, vmin=size[-2], vmax=size[-1], cmap="gist_earth_r", norm=norm, origin='lower')
    fig.show()

def test_extract_topo():
    module = LoadSaveTopoModule(extent=extent)
    module.frame = frame
    absolute_topo, relative_topo = module.extractTopo()
    #assert
    print(absolute_topo)

def test_save_topo():
    module = LoadSaveTopoModule(extent=extent)
    module.frame = frame
    module.extractTopo()
    module.saveTopo(filename=test_data['test']+'temp/01_savedTopo.npz')

def test_load_topo():
    module = LoadSaveTopoModule(extent=extent)
    module.loadTopo(filename=test_data['test']+'temp/01_savedTopo.npz')
    assert module.absolute_topo is not None
    print(module.file_id)

def test_release_area():
    module = LoadSaveTopoModule(extent=extent)
    module.add_release_area_origin(x=20, y=40)
    module.save_release_area(filename=test_data['test']+'temp/01_releaseArea.npy')

def test_show_loaded_topo():
    module = LoadSaveTopoModule(extent=extent)
    fig, ax = plt.subplots()
    module.box_width = 100
    module.box_height = 50
    module.frame = frame
    module.extractTopo()
    module.showLoadedTopo(ax)
    ax.imshow(frame, vmin=extent[-2], vmax=extent[-1], cmap="gist_earth_r", origin='lower')
    fig.show()

def test_extract_difference():
    module = LoadSaveTopoModule(extent=extent)
    module.frame = frame
    module.loadTopo()
    diff = module.extractDifference()
    print(diff)

def test_show_difference():
    module = LoadSaveTopoModule(extent=extent)
    fig, ax = plt.subplots()
    module.box_width = 100
    module.box_height = 50
    module.frame = frame
    module.extractTopo()
    module.box_origin = [10,10]
    module.showDifference(ax)
    ax.imshow(frame, vmin=extent[-2], vmax=extent[-1], cmap="gist_earth_r", origin='lower')
    fig.show()

def test_get_file_id():
    module = LoadSaveTopoModule(extent=extent)
    module._get_id(test_data['landslide_topo'] + 'Topography_1.npz')
    assert module.file_id == '1'

def test_snapshot_frame():
    module = LoadSaveTopoModule(extent=extent)
    module.box_width = 100
    module.box_height = 50
    module.frame = frame
    module.extractTopo()
    module.snapshotFrame()
    module.snapshot_frame.show()

def test_widgets():
    module = LoadSaveTopoModule(extent=extent)
    widgets = module.show_widgets()
    widgets.show()
