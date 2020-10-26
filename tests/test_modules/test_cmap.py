from sandbox import _test_data as test_data
from sandbox.projector import CmapModule
import numpy as np

file = np.load(test_data['topo'] + "DEM1.npz")
frame = file['arr_0']
extent = np.asarray([0, frame.shape[1], 0, frame.shape[0], frame.min(), frame.max()])

import matplotlib.pyplot as plt


def test_init():
    module = CmapModule(extent=extent)
    print(module)

def test_render_frame():
    module = CmapModule(extent=extent)
    fig, ax = plt.subplots()
    module.render_frame(frame, ax)
    fig.show()

def test_change_cmap():
    module = CmapModule(extent=extent)
    fig1, ax1 = plt.subplots()
    module.render_frame(frame, ax1)
    fig1.show()
    cmap = plt.cm.get_cmap('Accent_r')
    fig2, ax2 = plt.subplots()
    module.set_cmap(cmap)
    module.render_frame(frame, ax2)
    fig2.show()

def test_change_array():
    fig, ax = plt.subplots()
    col = ax.imshow(frame)
    fig.show()

    file = np.load(test_data['topo'] + "DEM2.npz")
    frame2 = file['arr_0']
    col.set_array(frame2)
    fig.show()

def test_change_cmap():
    fig, ax = plt.subplots()
    col = ax.imshow(frame)
    fig.show()

    cmap = plt.cm.get_cmap("hot")
    col.set_cmap(cmap)
    fig.show()

def test_delete_image():
    module = CmapModule(extent=extent)
    fig, ax = plt.subplots()
    module.render_frame(frame, ax)
    fig.show()

    module.delete_image()
    fig.show()

def test_update():
    module = CmapModule(extent=extent)
    fig, ax = plt.subplots()
    module.render_frame(frame, ax)
    sb_params = {'frame': frame,
                 'ax': ax,
                 'extent': extent,
                 'marker': [],
                 'cmap': plt.cm.get_cmap('viridis'),
                 'norm': None,
                 'active_cmap': True,
                 'active_contours': True}
    sb_params = module.update(sb_params)
    fig.show()

def test_widgets():
    module = CmapModule(extent=extent)
    widget = module.show_widgets()
    widget.show()
