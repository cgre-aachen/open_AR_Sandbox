from sandbox import _test_data as test_data
from sandbox.modules import GradientModule
import matplotlib.pyplot as plt
import pytest

import numpy as np
file = np.load(test_data['topo'] + "DEM1.npz")
frame = file['arr_0']
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
    module = GradientModule()
    print(module)

def test_update_dx():
    pytest.sb_params['ax'].cla()
    module = GradientModule(extent=extent)
    module.current_grad = module.grad_type[0]
    sb_params = module.update(pytest.sb_params)
    ax = sb_params['ax']
    fig = sb_params['fig']
    ax.imshow(sb_params.get('frame'), vmin=sb_params.get('extent')[-2], vmax=sb_params.get('extent')[-1],
              cmap=sb_params.get('cmap'), origin='lower')
    fig.show()


def test_update_dy():
    pytest.sb_params['ax'].cla()
    pytest.sb_params['frame']=frame
    module = GradientModule(extent=extent)
    module.current_grad = module.grad_type[1]
    sb_params = module.update(pytest.sb_params)
    ax = sb_params['ax']
    fig = sb_params['fig']
    ax.imshow(sb_params.get('frame'), vmin=sb_params.get('extent')[-2], vmax=sb_params.get('extent')[-1],
              cmap=sb_params.get('cmap'), origin='lower')
    fig.show()


def test_update_dxdy():
    pytest.sb_params['ax'].cla()
    module = GradientModule(extent=extent)
    module.current_grad = module.grad_type[2]
    sb_params = module.update(pytest.sb_params)
    ax = sb_params['ax']
    fig = sb_params['fig']
    ax.imshow(sb_params.get('frame'), vmin=sb_params.get('extent')[-2], vmax=sb_params.get('extent')[-1],
              cmap=sb_params.get('cmap'), origin='lower')
    fig.show()


def test_update_laplace():
    pytest.sb_params['ax'].cla()
    module = GradientModule(extent=extent)
    module.current_grad = module.grad_type[3]
    sb_params = module.update(pytest.sb_params)
    ax = sb_params['ax']
    fig = sb_params['fig']
    ax.imshow(sb_params.get('frame'), vmin=sb_params.get('extent')[-2], vmax=sb_params.get('extent')[-1],
              cmap=sb_params.get('cmap'), origin='lower')
    fig.show()


def test_update_vectorfield():
    pytest.sb_params['ax'].cla()
    module = GradientModule(extent=extent)
    module.current_grad = module.grad_type[5]
    sb_params = module.update(pytest.sb_params)
    ax = sb_params['ax']
    fig = sb_params['fig']
    ax.imshow(sb_params.get('frame'), vmin=sb_params.get('extent')[-2], vmax=sb_params.get('extent')[-1],
              cmap=sb_params.get('cmap'), origin='lower')
    fig.show()

def test_update_stream():
    pytest.sb_params['ax'].cla()
    module = GradientModule(extent=extent)
    module.current_grad = module.grad_type[6]
    sb_params = module.update(pytest.sb_params)
    ax = sb_params['ax']
    fig = sb_params['fig']
    ax.imshow(sb_params.get('frame'), vmin=sb_params.get('extent')[-2], vmax=sb_params.get('extent')[-1],
              cmap=sb_params.get('cmap'), origin='lower')
    fig.show()


def test_update_laplace_vectorfield():
    pytest.sb_params['ax'].cla()
    module = GradientModule(extent=extent)
    module.current_grad = module.grad_type[7]
    sb_params = module.update(pytest.sb_params)
    ax = sb_params['ax']
    fig = sb_params['fig']
    ax.imshow(sb_params.get('frame'), vmin=sb_params.get('extent')[-2], vmax=sb_params.get('extent')[-1],
              cmap=sb_params.get('cmap'), origin='lower')
    fig.show()


def test_update_laplace_stream():
    pytest.sb_params['ax'].cla()
    module = GradientModule(extent=extent)
    module.current_grad = module.grad_type[8]
    sb_params = module.update(pytest.sb_params)
    ax = sb_params['ax']
    fig = sb_params['fig']
    ax.imshow(sb_params.get('frame'), vmin=sb_params.get('extent')[-2], vmax=sb_params.get('extent')[-1],
              cmap=sb_params.get('cmap'), origin='lower')
    fig.show()


def test_update_lightsource():
    pytest.sb_params['ax'].cla()
    module = GradientModule(extent=extent)
    module.current_grad = module.grad_type[4]
    sb_params = module.update(pytest.sb_params)
    ax = sb_params['ax']
    fig = sb_params['fig']
    ax.imshow(sb_params.get('frame'), vmin=sb_params.get('extent')[-2], vmax=sb_params.get('extent')[-1],
              cmap=sb_params.get('cmap'), origin='lower')
    fig.show()

def test_modify_lightsource():
    pytest.sb_params['ax'].cla()
    module = GradientModule(extent=extent)
    module.set_lightsource(90, 30, 0.5)
    module.current_grad = module.grad_type[4]
    sb_params = module.update(pytest.sb_params)
    ax = sb_params['ax']
    fig = sb_params['fig']
    ax.imshow(sb_params.get('frame'), vmin=sb_params.get('extent')[-2], vmax=sb_params.get('extent')[-1],
              cmap=sb_params.get('cmap'), origin='lower')
    fig.show()

def test_widget():
    module = GradientModule(extent=extent)
    widgets = module.show_widgets()
    widgets.show()



