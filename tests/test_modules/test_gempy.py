from sandbox import _test_data as test_data
from sandbox.modules import GemPyModule
import matplotlib.pyplot as plt

import numpy as np
file = np.load(test_data['topo'] + "DEM1.npz")
frame = file['arr_0'] + 500 #assuming the mean of the previous one was 1000
extent = [0, frame.shape[1], 0, frame.shape[0], frame.min(), frame.max()]

import os
import gempy as gp
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN"


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

def create_example_model(name, extent=[0, 1000, 0, 1000, 0, 1000], do_sections=False,
                         change_color=False, data_path=test_data['gempy_data'], theano_optimizer='fast_compile'):
    all_models = ['Horizontal_layers', 'Recumbent_fold', 'Anticline',
                  'Pinchout', 'Fault', 'Unconformity']


    assert name in all_models, 'possible model names are ' + str(all_models)
    geo_model = gp.create_model(name)

    if name == 'Horizontal_layers':
        geo_model = gp.init_data(geo_model, extent=extent, resolution=[2, 2, 2],
                                 path_o=data_path + "model1_orientations.csv",
                                 path_i=data_path + "model1_surface_points.csv")
        if change_color:
            geo_model.surfaces.colors.change_colors({"rock2": '#9f0052', 'rock1': '#e36746',
                                                     'basement': '#f9f871'})

        gp.map_series_to_surfaces(geo_model, {"Strat_Series": ('rock2', 'rock1'),
                                              "Basement_Series": ('basement')})

    elif name == 'Recumbent_fold':
        geo_model = gp.init_data(geo_model, extent=extent, resolution=[2, 2, 2],
                                 path_o=data_path + "model3_orientations.csv",
                                 path_i=data_path + "model3_surface_points.csv")
        gp.map_series_to_surfaces(geo_model, {"Strat_Series": ('rock2', 'rock1'),
                                              "Basement_Series": ('basement')})
        if change_color:
            geo_model.surfaces.colors.change_colors({"rock2": '#e36746', 'rock1': '#c0539f',
                                                     'basement': '#006fa8'})

    elif name == 'Anticline':
        geo_model = gp.init_data(geo_model, extent=extent, resolution=[2, 2, 2],
                                 path_o=data_path + "model2_orientations.csv",
                                 path_i=data_path + "model2_surface_points.csv")
        gp.map_series_to_surfaces(geo_model, {"Strat_Series": ('rock2', 'rock1'),
                                              "Basement_Series": ('basement')})

    elif name == 'Pinchout':
        geo_model = gp.init_data(geo_model, extent=extent, resolution=[2, 2, 2],
                                 path_o=data_path + "model4_orientations.csv",
                                 path_i=data_path + "model4_surface_points.csv")
        gp.map_series_to_surfaces(geo_model, {"Strat_Series": ('rock2', 'rock1'),
                                              "Basement_Series": ('basement')})
        if change_color:
            geo_model.surfaces.colors.change_colors({"rock2": '#a1b455', 'rock1': '#ffbe00',
                                                     'basement': '#006471'})

    elif name == 'Fault':
        geo_model = gp.init_data(geo_model, extent=extent, resolution=[2, 2, 2],
                                 path_o=data_path + "model5_orientations.csv",
                                 path_i=data_path + "model5_surface_points.csv")
        gp.map_series_to_surfaces(geo_model, {"Fault_Series": 'fault',
                                              "Strat_Series": ('rock2', 'rock1')})
        geo_model.set_is_fault(['Fault_Series'], change_color=False)
        if change_color:
            geo_model.surfaces.colors.change_colors({"rock2": '#00c2d0', 'rock1': '#a43d00',
                                                     'basement': '#76a237', 'fault': '#000000'})

    elif name == 'Unconformity':
        geo_model = gp.init_data(geo_model, extent=extent, resolution=[2, 2, 2],
                                 path_o=data_path + "model6_orientations.csv",
                                 path_i=data_path + "model6_surface_points.csv")

        gp.map_series_to_surfaces(geo_model, {"Strat_Series1": ('rock3'),
                                              "Strat_Series2": ('rock2', 'rock1'),
                                              "Basement_Series": ('basement')})

    if do_sections:
        geo_model.grid.set_section_grid({'section' + ' ' + name: ([0, 500], [1000, 500], [30, 30])})

    interp_data = gp.set_interpolator(geo_model, compile_theano=True,
                                      theano_optimizer=theano_optimizer)

    _ = gp.compute_model(geo_model, compute_mesh=False)

    if do_sections:
        gp.plot.plot_section_by_name(geo_model, 'section' + ' ' + name, show_data=False)

    return geo_model

def test_create_model():
    geo_model = create_example_model(name = 'Horizontal_layers')
    print(geo_model)

def test_init():
    geo_model = create_example_model('Horizontal_layers')
    module = GemPyModule(geo_model = geo_model, extent=extent, box=[1000, 800])
    print(module)

def test_update():
    geo_model = create_example_model('Horizontal_layers')
    module = GemPyModule(geo_model = geo_model, extent=extent, box=[1000, 800])
    fig, ax = plt.subplots()
    depth, ax, size, cmap, norm = module.update(frame, ax, extent)
    fig.show()

def test_change_model():
    geo_model = create_example_model('Horizontal_layers')
    module = GemPyModule(geo_model = geo_model, extent=extent, box=[1000, 800])
    fig, ax = plt.subplots()
    depth, ax, size, cmap, norm = module.update(frame, ax, extent)
    fig.show()

    geo_model2 = create_example_model('Fault')
    module.change_model(geo_model2)
    depth, ax, size, cmap, norm = module.update(frame, ax, extent)
    fig.show()

def test_section_dictionaries_cross_section():
    geo_model = create_example_model('Horizontal_layers')
    module = GemPyModule(geo_model=geo_model, extent=extent, box=[1000, 800])
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
    module = GemPyModule(geo_model=geo_model, extent=extent, box=[1000, 800])
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
    module = GemPyModule(geo_model=geo_model, extent=extent, box=[1000, 800])
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
    module = GemPyModule(geo_model=geo_model, extent=extent, box=[1000, 800])
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
    module = GemPyModule(geo_model=geo_model, extent=extent, box=[1000, 800])
    module.setup(frame)

    module.set_borehole_dict((10, 20), "borehole1")
    module.set_borehole_dict((500, 500), "borehole2")

    module._get_polygon_data()

    print(module.borehole_tube, module.colors_bh)

def test_plot_boreholes():
    geo_model = create_example_model('Horizontal_layers')
    module = GemPyModule(geo_model=geo_model, extent=extent, box=[1000, 800])
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

    module.plot_boreholes()