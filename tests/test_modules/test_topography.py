from sandbox import _test_data as test_data
from sandbox.modules import TopoModule
import matplotlib.pyplot as plt
import pytest
import numpy as np

file = np.load(test_data['topo'] + "DEM1.npz")
frame = file['arr_0']
frame = frame + np.abs(frame.min())
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
    sb_params = module.update(pytest.sb_params)
    ax = sb_params['ax']
    fig = sb_params['fig']
    ax.imshow(sb_params.get('frame'), #vmin=sb_params.get('extent')[-2], vmax=sb_params.get('extent')[-1],
              cmap=sb_params.get('cmap'), norm=sb_params.get('norm'), origin='lower')
    fig.show()
    pytest.sb_params['frame'] = frame
    module.sea = True
    module.sea_contour = True
    sb_params = module.update(pytest.sb_params)
    ax = sb_params['ax']
    fig = sb_params['fig']
    ax.imshow(sb_params.get('frame'), #vmin=sb_params.get('extent')[-2], vmax=sb_params.get('extent')[-1],
              cmap=sb_params.get('cmap'), norm=sb_params.get('norm'), origin='lower')
    fig.show()
    pytest.sb_params['ax'].cla()
    module.sea_contour = False
    pytest.sb_params['frame'] = frame
    sb_params = module.update(pytest.sb_params)
    ax = sb_params['ax']
    fig = sb_params['fig']
    ax.imshow(sb_params.get('frame'), #vmin=sb_params.get('extent')[-2], vmax=sb_params.get('extent')[-1],
              cmap=sb_params.get('cmap'), norm=sb_params.get('norm'), origin='lower')
    fig.show()

def test_normalize_negative_height():
    module = TopoModule(extent=extent)
    plt.imshow(pytest.sb_params['frame'], origin="lower", cmap = "viridis")
    plt.colorbar()
    plt.show()
    n = -200
    new_frame_negative, new_extent_negative = module.normalize_topography(pytest.sb_params['frame'].copy(),
                                                                          pytest.sb_params['extent'].copy(),
                                                                          max_height=1000, min_height= n)
    assert new_frame_negative.min() == n
    plt.imshow(new_frame_negative, origin="lower", cmap="viridis")
    plt.colorbar()
    plt.show()
    p=200
    new_frame_positive, new_extent_positive = module.normalize_topography(pytest.sb_params['frame'].copy(),
                                                                          pytest.sb_params['extent'].copy(),
                                                                          max_height=1000, min_height=p)
    assert new_frame_positive.min() == p
    plt.imshow(new_frame_positive, origin="lower", cmap="viridis")
    plt.colorbar()
    plt.show()

    c = 0
    new_frame_positive, new_extent_positive = module.normalize_topography(pytest.sb_params['frame'].copy(),
                                                                          pytest.sb_params['extent'].copy(),
                                                                          max_height=1000, min_height=c)
    assert new_frame_positive.min() == c
    plt.imshow(new_frame_positive, origin="lower", cmap="viridis")
    plt.colorbar()
    plt.show()