from sandbox import _test_data as test_data
from sandbox.modules import TopoModule
import matplotlib.pyplot as plt

import numpy as np
file = np.load(test_data['topo'] + "DEM1.npz")
frame = file['arr_0']
extent = np.asarray([0, frame.shape[1], 0, frame.shape[0], frame.min(), frame.max()])

def test_init():
    fig, ax = plt.subplots()
    module = TopoModule()
    print(module)
    
def test_update():
    module = TopoModule(extent=extent)
    fig, ax = plt.subplots()
    depth, ax, size, cmap, norm = module.update(frame, ax, extent, box = None)
    ax.imshow(depth, vmin=size[-2], vmax=size[-1], cmap=cmap, norm=norm, origin='lower')
    fig.show()

def test_update_no_see():
    module = TopoModule(extent=extent)
    fig, ax = plt.subplots()
    module.see = False
    depth, ax, size, cmap, norm = module.update(frame, ax, extent)
    ax.imshow(depth, vmin=size[-2], vmax=size[-1], cmap=cmap, norm=norm, origin='lower')
    fig.show()

def test_update_no_normalized():
    module = TopoModule(extent=extent)
    fig, ax = plt.subplots()
    module.normalize = False
    depth, ax, size, cmap, norm = module.update(frame, ax, extent)
    ax.imshow(depth, vmin=size[-2], vmax=size[-1], cmap=cmap, norm=norm, origin='lower')
    fig.show()

def test_widgets():
    module = TopoModule(extent=extent)
    widgets = module.widgets()
    widgets.show()



