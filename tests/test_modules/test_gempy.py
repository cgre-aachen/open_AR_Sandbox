import os
import gempy as gp
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN"
from sandbox import _test_data as test_data
from sandbox.modules import GemPyModule
from sandbox.modules.gempy.example_models import *
import matplotlib.pyplot as plt

import numpy as np
#with np.load(test_data['topo'] + "DEM1.npz") as data:
#    file = data
file = np.load(file=test_data['topo'] + "DEM4.npz", allow_pickle=True)#, encoding='bytes', allow_pickle=False)
frame = file['arr_0'] + 500 #assuming the mean of the previous one was 1000
extent = [0, frame.shape[1], 0, frame.shape[0], frame.min(), frame.max()]

def test_scale():
    from sandbox.modules.gempy.utils import get_scale
    sca = get_scale(physical_extent=[1000, 800],
                model_extent=[0, 1000, 0, 1000, 0, 2300],
                sensor_extent=extent)
    print(sca)


def test_grid():
    from sandbox.modules.gempy.utils import Grid
    grid = Grid(physical_extent=[1000, 800],
                model_extent=[0, 1000, 0, 1000, 0, 2300],
                sensor_extent=extent,
                scale=None)
    grid.update_grid(frame)
    print(grid.depth_grid)

def test_create_model():
    geo_model = create_example_model(name = 'Horizontal_layers')
    print(geo_model)

def test_create_model_predefined():
    model_dict = create_model_dict(all_models)
    print(model_dict)

def test_init_examples():
    module = GemPyModule(geo_model=None, extent=extent, box=[1000, 800], load_examples=True)
    print(module)
    print(module.model_dict.keys())

def test_load_geomodel_from_examples():
    module = GemPyModule(geo_model=None, extent=extent, box=[1000, 800],
                         load_examples=True, name_example=["Anticline", "Horizontal_layers"])
    print(module.geo_model)
    print(module.model_dict.keys())

def test_init():
    geo_model = create_example_model('Horizontal_layers')
    module = GemPyModule(geo_model = geo_model, extent=extent, box=[1000, 800], load_examples=False)
    print(module)

def test_update():
    geo_model = create_example_model('Horizontal_layers')
    module = GemPyModule(geo_model = geo_model, extent=extent, box=[1000, 800], load_examples=False)
    fig, ax = plt.subplots()
    depth, ax, size, cmap, norm = module.update(frame, ax, extent)
    fig.show()

def test_change_model():
    geo_model = create_example_model('Horizontal_layers')
    module = GemPyModule(geo_model = geo_model, extent=extent, box=[1000, 800], load_examples=False)
    fig, ax = plt.subplots()
    depth, ax, size, cmap, norm = module.update(frame, ax, extent)
    fig.show()

    geo_model2 = create_example_model('Fault')
    module.change_model(geo_model2)
    depth, ax, size, cmap, norm = module.update(frame, ax, extent)
    fig.show()

def test_section_dictionaries_cross_section():
    geo_model = create_example_model('Horizontal_layers')
    module = GemPyModule(geo_model=geo_model, extent=extent, box=[1000, 800], load_examples=False)
    module.setup(frame)
    module.set_section_dict((10, 10), (500, 500), "Section1")
    module.set_section_dict((100, 100), (500, 10), "Section2")
    module.show_actual_model()

    assert module.section_dict == {'Section1': ([10, 10], [500, 500], [150, 100]),
                                   'Section2': ([100, 100], [500, 10], [150, 100]),
                                   'Model: Horizontal_layers': ([0.0, 500.0], [1000.0, 500.0], [150, 100])}

    module.remove_section_dict("Section2")

    assert module.section_dict == {'Section1': ([10, 10], [500, 500], [150, 100]),
                                   'Model: Horizontal_layers': ([0.0, 500.0], [1000.0, 500.0], [150, 100])}

def test_plot_cross_sections():
    geo_model = create_example_model('Horizontal_layers')
    module = GemPyModule(geo_model=geo_model, extent=extent, box=[1000, 800], load_examples=False)
    fig, ax = plt.subplots()
    depth, ax, size, cmap, norm = module.update(frame, ax, extent)
    module.set_section_dict((10, 10), (500, 500), "Section1")
    module.set_section_dict((100, 100), (500, 10), "Section2")
    module.show_actual_model()
    module.show_section_traces()
    module.show_cross_section("Section1")
    module.show_cross_section("Section2")
    module.show_geological_map()

def test_show_plot_widgets():
    geo_model = create_example_model('Horizontal_layers')
    module = GemPyModule(geo_model=geo_model, extent=extent, box=[1000, 800], load_examples=False)
    fig, ax = plt.subplots()
    depth, ax, size, cmap, norm = module.update(frame, ax, extent)
    module.set_section_dict((10, 10), (500, 500), "Section1")
    module.set_section_dict((100, 100), (500, 10), "Section2")
    module.show_actual_model()

    module.show_section_traces()

    module.show_cross_section("Section1")

    module.show_geological_map()

    module.panel_actual_model.show()
    module.panel_plot_2d.show()
    module.panel_section_traces.show()
    module.panel_geo_map.show()

def test_borehole_dict():
    geo_model = create_example_model('Horizontal_layers')
    module = GemPyModule(geo_model=geo_model, extent=extent, box=[1000, 800], load_examples=False)
    module.setup(frame)

    module.set_borehole_dict((10,20), "borehole1")
    module.set_borehole_dict((500, 500), "borehole2")

    print(module.borehole_dict)

    assert module.borehole_dict == {'borehole1': ([10, 20], [11, 20], [5, 5]),
                                   'borehole2': ([500, 500], [501, 500], [5, 5])}

    module.remove_borehole_dict("borehole2")

    print(module.borehole_dict)

    assert module.borehole_dict == {'borehole1': ([10, 20], [11, 20], [5, 5])}

def test_polygon_data_boreholes():
    geo_model = create_example_model('Horizontal_layers')
    module = GemPyModule(geo_model=geo_model, extent=extent, box=[1000, 800], load_examples=False)
    module.setup(frame)

    module.set_borehole_dict((10, 20), "borehole1")
    module.set_borehole_dict((500, 500), "borehole2")

    module._get_polygon_data()

    print(module.borehole_tube, module.colors_bh)

def test_plot_boreholes():
    #geo_model = create_example_model('Fault')
    geo_model = create_example_model('Horizontal_layers')
    module = GemPyModule(geo_model=geo_model, extent=extent, box=[1000, 800], load_examples=False)
    module.setup(frame)

    module.set_borehole_dict((10, 20), "borehole1")
    module.set_borehole_dict((200, 500), "borehole2")
    module.set_borehole_dict((500, 500), "borehole3")
    module.set_borehole_dict((900, 500), "borehole4")
    module.set_borehole_dict((100, 100), "borehole5")
    module.set_borehole_dict((600, 700), "borehole6")
    module.set_borehole_dict((200, 800), "borehole7")
    module.set_borehole_dict((800, 200), "borehole8")

    module._get_polygon_data()

    p = module.plot_boreholes(notebook = False)
    p.show()

def test_update_borehole_panel():
    geo_model = create_example_model('Horizontal_layers')
    module = GemPyModule(geo_model=geo_model, extent=extent, box=[1000, 800], load_examples=False)
    module.setup(frame)

    module.set_borehole_dict((10, 20), "borehole1")
    module.set_borehole_dict((200, 500), "borehole2")
    module.set_borehole_dict((500, 500), "borehole3")
    module.set_borehole_dict((900, 500), "borehole4")
    module.set_borehole_dict((100, 100), "borehole5")
    module.set_borehole_dict((600, 700), "borehole6")
    module.set_borehole_dict((200, 800), "borehole7")
    module.set_borehole_dict((800, 200), "borehole8")

    module._get_polygon_data()

    vtk = module.show_boreholes_panel()
    vtk.show()

def test_widgets():
    module = GemPyModule(geo_model=None, extent=extent, box=[1000, 800], load_examples=True)
    module.setup(frame)

    fig, ax = plt.subplots()
    depth, ax, size, cmap, norm = module.update(frame, ax, extent)

    module.set_section_dict((10, 10), (500, 500), "Section1")
    module.set_section_dict((100, 100), (500, 10), "Section2")

    module.set_borehole_dict((500, 500), "borehole3")
    module.set_borehole_dict((900, 500), "borehole4")

    widgets = module.show_widgets()

    widgets.show()




