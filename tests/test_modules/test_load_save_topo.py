from sandbox import _test_data as test_data
from sandbox.modules import LoadSaveTopoModule
import matplotlib.pyplot as plt
import pytest
import numpy as np

file = np.load(test_data['topo'] + "DEM1.npz")
frame = file['arr_0']
extent = [0, frame.shape[1], 0, frame.shape[0], frame.min(), frame.max()]


def load_marker():
    import pandas as pd
    from sandbox import _test_data
    arucos = _test_data['test'] + "arucos.pkl"
    try:
        df = pd.read_pickle(arucos)
        print("Arucos loaded")
    except:
        df = pd.DataFrame()
        print("No arucos found")
    return df

fig, ax = plt.subplots()
pytest.sb_params = {'frame': frame,
                    'ax': ax,
                    'fig': fig,
                    'extent': extent,
                    'marker': load_marker(),
                    'cmap': plt.cm.get_cmap('gist_earth_r'),
                    'norm': None,
                    'active_cmap': True,
                    'active_contours': True}

def update(module):
    pytest.sb_params['ax'].cla()
    sb_params = module.update(pytest.sb_params)
    ax = sb_params['ax']
    fig = sb_params['fig']
    ax.imshow(sb_params.get('frame'), vmin=sb_params.get('extent')[-2], vmax=sb_params.get('extent')[-1],
              cmap=sb_params.get('cmap'), norm=sb_params.get('norm'), origin='lower')
    fig.show()


def test_init():
    module = LoadSaveTopoModule(extent=extent)
    print(module)

def test_update():
    pytest.sb_params['ax'].cla()
    module = LoadSaveTopoModule(extent=extent)
    module.box_width = 100
    module.box_height = 50
    update(module)

def test_plot_release_area():
    module = LoadSaveTopoModule(extent=extent)
    module.box_width = 100
    module.box_height = 50
    module.add_release_area_origin(x=20, y=40)
    module.add_release_area_origin(x=50, y=40)
    module.add_release_area_origin(x=100, y=90)
    update(module)

def test_extract_topo():
    module = LoadSaveTopoModule(extent=extent)
    module.frame = frame
    absolute_topo, relative_topo = module.extractTopo()
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
    module.loadTopo(filename=test_data['test']+'temp/01_savedTopo.npz')
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
