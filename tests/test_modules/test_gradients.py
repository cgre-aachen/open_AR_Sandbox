from sandbox import _test_data as test_data
from sandbox.modules import GradientModule
import matplotlib.pyplot as plt

import numpy as np
file = np.load(test_data['topo'] + "DEM1.npz")
frame = file['arr_0']
extent = [0, frame.shape[1], 0, frame.shape[0], frame.min(), frame.max()]


def test_init():
    module = GradientModule()
    print(module)


def test_update_dx():
    module = GradientModule(extent=extent)
    fig, ax = plt.subplots()
    module.current_grad = module.grad_type[0]
    depth, ax, size, cmap, norm = module.update(frame, ax, extent)
    ax.imshow(depth, vmin=size[-2], vmax=size[-1], cmap=cmap, norm=norm, origin='lower')
    fig.show()


def test_update_dy():
    module = GradientModule(extent=extent)
    fig, ax = plt.subplots()
    module.current_grad = module.grad_type[1]
    depth, ax, size, cmap, norm = module.update(frame, ax, extent)
    ax.imshow(depth, vmin=size[-2], vmax=size[-1], cmap=cmap, norm=norm, origin='lower')
    fig.show()


def test_update_dxdy():
    module = GradientModule(extent=extent)
    fig, ax = plt.subplots()
    module.current_grad = module.grad_type[2]
    depth, ax, size, cmap, norm = module.update(frame, ax, extent)
    ax.imshow(depth, vmin=size[-2], vmax=size[-1], cmap=cmap, norm=norm, origin='lower')
    fig.show()


def test_update_laplace():
    module = GradientModule(extent=extent)
    fig, ax = plt.subplots()
    module.current_grad = module.grad_type[3]
    depth, ax, size, cmap, norm = module.update(frame, ax, extent)
    ax.imshow(depth, vmin=size[-2], vmax=size[-1], cmap=cmap, norm=norm, origin='lower')
    fig.show()


def test_update_lightsource():
    module = GradientModule(extent=extent)
    fig, ax = plt.subplots()
    module.current_grad = module.grad_type[4]
    depth, ax, size, cmap, norm = module.update(frame, ax, extent)
    ax.imshow(depth, vmin=size[-2], vmax=size[-1], cmap=cmap, norm=norm, origin='lower')
    fig.show()


def test_modify_lightsource():
    module = GradientModule(extent=extent)
    fig, ax = plt.subplots()
    module.current_grad = module.grad_type[4]
    depth, ax, size, cmap, norm = module.update(frame, ax, extent)
    ax.imshow(depth, vmin=size[-2], vmax=size[-1], cmap=cmap, norm=norm, origin='lower')
    fig.show()

    module2 = GradientModule(extent=extent)
    fig2, ax2 = plt.subplots()
    module2.set_lightsource(90, 30, 0.5)
    module2.current_grad = module.grad_type[4]
    depth2, ax2, size2, cmap2, norm2 = module2.update(frame, ax2, extent)
    ax2.imshow(depth2, vmin=size[-2], vmax=size[-1], cmap=cmap2, norm=norm2, origin='lower')
    fig2.show()

def test_update_vectorfield():
    module = GradientModule(extent=extent)
    fig, ax = plt.subplots()
    module.current_grad = module.grad_type[5]
    module.update(frame, ax, extent)
    fig.show()


def test_update_stream():
    module = GradientModule(extent=extent)
    fig, ax = plt.subplots()
    module.current_grad = module.grad_type[6]
    module.update(frame, ax, extent)
    fig.show()


def test_update_laplace_vectorfield():
    module = GradientModule(extent=extent)
    fig, ax = plt.subplots()
    module.current_grad = module.grad_type[7]
    depth, ax, size, cmap, norm = module.update(frame, ax, extent)
    ax.imshow(depth, vmin=size[-2], vmax=size[-1], cmap=cmap, norm=norm, origin='lower')
    fig.show()


def test_update_laplace_stream():
    module = GradientModule(extent=extent)
    fig, ax = plt.subplots()
    module.current_grad = module.grad_type[8]
    depth, ax, size, cmap, norm = module.update(frame, ax, extent)
    ax.imshow(depth, vmin=size[-2], vmax=size[-1], cmap=cmap, norm=norm, origin='lower')
    fig.show()


def test_widget():
    module = GradientModule(extent=extent)
    widgets = module.show_widgets()
    widgets.show()
