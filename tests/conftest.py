import matplotlib.pyplot as plt
import pytest
import numpy as np
import pandas as pd

@pytest.fixture()
def data_files():
    from sandbox import _test_data
    return _test_data

@pytest.fixture()
def sb_params(data_files):
    df = pd.read_pickle(data_files['test']+"arucos.pkl")
    file = np.load(data_files['topo'] + "DEM1.npz")
    frame = file['arr_0']
    frame = frame + np.abs(np.amin(frame))
    extent = [0, frame.shape[1], 0, frame.shape[0], frame.min(), frame.max()]

    fig, ax = plt.subplots()
    return  {'frame': frame,
                      'ax': ax,
                        'fig': fig,
                      'set_colorbar': None, #self.projector.set_colorbar,
                      'set_legend': None, #self.projector.set_legend,
                      'extent': extent,
                      'box_dimensions': None, #self.sensor.physical_dimensions,
                      'marker': df,
                      'cmap': plt.cm.get_cmap('gist_earth'),
                      'norm': None,
                      'active_cmap': True,
                      'active_shading': True,
                      'active_contours': True,
                      'same_frame': False,
                      'lock_thread': None, #self.lock,
                      'trigger': None, #self.projector.trigger,
                      'del_contour': True, }




