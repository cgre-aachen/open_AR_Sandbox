from sandbox import _test_data as test_data
from sandbox.modules import TopoModule
import matplotlib.pyplot as plt
import pytest
import numpy as np

file = np.load(test_data['topo'] + "DEM1.npz")
frame = file['arr_0']
extent = np.asarray([0, frame.shape[1], 0, frame.shape[0], frame.min(), frame.max()])

fig, ax = plt.subplots()
pytest.sb_params = {'frame': frame,
                    'ax': ax,
                    'fig': fig,
                    'extent': extent,
                    'marker': [],
                    'cmap': plt.cm.get_cmap('gist_earth_r'),
                    'norm': None,
                    'active_cmap': True,
                    'active_contours': True}


def test_init():
    module = TopoModule()
    print(module)


def test_update():
    pytest.sb_params['ax'].cla()
    module = TopoModule(extent=extent)
    sb_params = module.update(pytest.sb_params)
    ax = sb_params['ax']
    fig = sb_params['fig']
    ax.imshow(sb_params.get('frame'), vmin=sb_params.get('extent')[-2], vmax=sb_params.get('extent')[-1],
              cmap=sb_params.get('cmap'), norm=sb_params.get('norm'), origin='lower')
    fig.show()


def test_update_no_sea():
    pytest.sb_params['ax'].cla()
    module = TopoModule(extent=extent)
    module.sea = False
    sb_params = module.update(pytest.sb_params)
    ax = sb_params['ax']
    fig = sb_params['fig']
    ax.imshow(sb_params.get('frame'), vmin=sb_params.get('extent')[-2], vmax=sb_params.get('extent')[-1],
              cmap=sb_params.get('cmap'), norm=sb_params.get('norm'), origin='lower')
    fig.show()


def test_update_no_normalized():
    pytest.sb_params['ax'].cla()
    module = TopoModule(extent=extent)
    module.normalize = False
    sb_params = module.update(pytest.sb_params)
    ax = sb_params['ax']
    fig = sb_params['fig']
    ax.imshow(sb_params.get('frame'), vmin=sb_params.get('extent')[-2], vmax=sb_params.get('extent')[-1],
              cmap=sb_params.get('cmap'), norm=sb_params.get('norm'), origin='lower')
    fig.show()


def test_widgets():
    module = TopoModule(extent=extent)
    widgets = module.show_widgets()
    widgets.show()


def test_add_countour():
    pytest.sb_params['ax'].cla()
    module = TopoModule(extent=extent)
    module.sea_contour = True
    sb_params = module.update(pytest.sb_params)
    ax = sb_params['ax']
    fig = sb_params['fig']
    ax.imshow(sb_params.get('frame'), vmin=sb_params.get('extent')[-2], vmax=sb_params.get('extent')[-1],
              cmap=sb_params.get('cmap'), norm=sb_params.get('norm'), origin='lower')
    fig.show()
