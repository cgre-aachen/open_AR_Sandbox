import os
import gempy as gp
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN"
from sandbox import _test_data as test_data
from sandbox.modules import GemPyModule
from sandbox.modules.gempy.example_models import *
import matplotlib.pyplot as plt
import pytest
import numpy as np

file = np.load(file=test_data['topo'] + "DEM4.npz", allow_pickle=True)#, encoding='bytes', allow_pickle=False)
frame = file['arr_0'] + 500 #assuming the mean of the previous one was 1000
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
    sb_params = module.update(pytest.sb_params)
    sb_params['fig'].show()

def test_change_model():
    geo_model = create_example_model('Horizontal_layers')
    module = GemPyModule(geo_model = geo_model, extent=extent, box=[1000, 800], load_examples=False)
    sb_params = module.update(pytest.sb_params)
    sb_params['fig'].show()

    geo_model2 = create_example_model('Fault')
    module.change_model(geo_model2)
    sb_params = module.update(pytest.sb_params)
    sb_params['fig'].show()

def test_section_dictionaries_cross_section():
    geo_model = create_example_model('Horizontal_layers')
    module = GemPyModule(geo_model=geo_model, extent=extent, box=[1000, 800], load_examples=False)
    module.setup(frame)
    module.set_section_dict((10, 10), (500, 500), "Section1")
    module.set_section_dict((100, 100), (500, 10), "Section2")
    module.show_actual_model()

    assert module.section_dict == {'Model: Horizontal_layers': ([0.0, 500.0], [1000.0, 500.0], [150, 100]),
                                   'Section1': ([10, 10], [500, 500], [150, 100]),
                                   'Section2': ([100, 100], [500, 10], [150, 100]),
                                   }

    module.remove_section_dict("Section2")

    assert module.section_dict == {'Model: Horizontal_layers': ([0.0, 500.0], [1000.0, 500.0], [150, 100]),
                                   'Section1': ([10, 10], [500, 500], [150, 100])}

def test_plot_cross_sections():
    geo_model = create_example_model('Horizontal_layers')
    module = GemPyModule(geo_model=geo_model, extent=extent, box=[1000, 800], load_examples=False)
    sb_params = module.update(pytest.sb_params)
    sb_params['fig'].show()
    module.set_section_dict((10, 10), (500, 500), "Section1")
    module.set_section_dict((100, 100), (500, 10), "Section2")
    module.show_actual_model()
    module.show_section_traces()
    module.show_cross_section("Section1")
    module.show_cross_section("Section2")
    module.show_geological_map()

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
    module.set_borehole_dict((200, 150), "borehole7")
    module.set_borehole_dict((150, 200), "borehole8")

    module._get_polygon_data()

    p = module.plot_boreholes(notebook=False, background=False)
    p.show()

def test_compute_model_space_arucos():
    from sandbox.sensor import Sensor
    from sandbox.markers import MarkerDetection
    from sandbox import _calibration_dir, _test_data
    sensor = Sensor(calibsensor=_calibration_dir + 'sensorcalib.json', name='kinect_v2')
    aruco = MarkerDetection(sensor=sensor)
    color = np.load(_test_data['test'] + 'frame1.npz')['arr_1']
    module = GemPyModule(geo_model=None, extent=sensor.extent, box=[1000, 800], load_examples=True,
                         name_example=['Horizontal_layers'])
    module.setup(sensor.get_frame())
    df = aruco.update(frame=color)
    df_new = module._compute_modelspace_arucos(df)
    print(df_new)

def test_set_aruco_dict():
    from sandbox.sensor import Sensor
    from sandbox.markers import MarkerDetection
    from sandbox import _calibration_dir, _test_data
    sensor = Sensor(calibsensor=_calibration_dir + 'sensorcalib.json', name='kinect_v2')
    aruco = MarkerDetection(sensor=sensor)
    color = np.load(_test_data['test'] + 'frame1.npz')['arr_1']
    module = GemPyModule(geo_model=None, extent=sensor.extent, box=[1000, 800], load_examples=True,
                         name_example=['Horizontal_layers'])
    module.setup(sensor.get_frame())
    df = aruco.update(frame=color)
    df_new = module._compute_modelspace_arucos(df)
    module.set_aruco_dict(df_new)

    print(module.model_sections_dict)

def test_update_arucos():
    from sandbox.sensor import Sensor
    from sandbox.markers import MarkerDetection
    from sandbox import _calibration_dir, _test_data
    sensor = Sensor(calibsensor=_calibration_dir + 'sensorcalib.json', name='kinect_v2', invert=False)
    aruco = MarkerDetection(sensor=sensor)
    color = np.load(_test_data['test'] + 'frame1.npz')['arr_1']
    module = GemPyModule(geo_model=None, extent=sensor.extent, box=[1000, 800], load_examples=True,
                         name_example=['Horizontal_layers'])
    sb_params = pytest.sb_params
    sb_params['frame'] = sensor.get_frame()
    module.setup(sb_params['frame'])
    sb_params['marker']= aruco.update(frame=color)

    sb_params = module.update(sb_params)
    sb_params['fig'].show()
    aruco.plot_aruco(sb_params['ax'], sb_params['marker'])
    fig.show()

def test_show_plot_widgets():
    geo_model = create_example_model('Horizontal_layers')
    module = GemPyModule(geo_model=geo_model, extent=extent, box=[1000, 800], load_examples=False)
    sb_params = module.update(pytest.sb_params)
    sb_params['fig'].show()
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

def test_3d_model_plot():
    module = GemPyModule(geo_model=None, extent=extent, box=[1000, 800], load_examples=True,
                         name_example=['Horizontal_layers'])
    module.setup(frame)
    geo_3d = module.plot_3d_model()
    geo_3d.p.show()

def test_3d_model_plot_widget():
    module = GemPyModule(geo_model=None, extent=extent, box=[1000, 800], load_examples=True,
                         name_example=['Horizontal_layers'])
    module.setup(frame)
    #module._plotter_type = 'background'
    vtk = module.show_3d_model_panel()
    vtk.show()

def test_3d_widgets():
    module = GemPyModule(geo_model=None, extent=extent, box=[1000, 800], load_examples=True,
                         name_example=['Horizontal_layers'])
    module.setup(frame)
    widget = module.widget_3d_model()
    widget.show()

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

    #fig, ax = plt.subplots()
    #depth, ax, size, cmap, norm, df = module.update(frame, ax, extent)
    module.setup(frame)

    module.set_section_dict((10, 10), (500, 500), "Section1")
    module.set_section_dict((100, 100), (500, 10), "Section2")

    module.set_borehole_dict((500, 500), "borehole3")
    module.set_borehole_dict((900, 500), "borehole4")

    widgets = module.show_widgets()

    widgets.show()

def test_cross_section_widgets():
    geo_model = create_example_model('Horizontal_layers')
    module = GemPyModule(geo_model=geo_model, extent=extent, box=[1000, 800], load_examples=False)
    module.setup(frame)
    module.set_section_dict((10, 10), (500, 500), "Section1")
    module.set_section_dict((100, 100), (500, 10), "Section2")
    widget=module.widget_cross_sections()
    widget.show()

def test_borehole_widgets():
    geo_model = create_example_model('Horizontal_layers')
    module = GemPyModule(geo_model=geo_model, extent=extent, box=[1000, 800], load_examples=False)
    module.setup(frame)
    module.set_borehole_dict((500, 500), "borehole3")
    module.set_borehole_dict((900, 500), "borehole4")
    module._get_polygon_data()
    widget = module.widget_boreholes()
    widget.show()

def test_widgets_with_arucos():
    from sandbox.sensor import Sensor
    from sandbox.markers import MarkerDetection
    from sandbox import _calibration_dir, _test_data
    sensor = Sensor(calibsensor=_calibration_dir + 'sensorcalib.json', name='kinect_v2')
    aruco = MarkerDetection(sensor=sensor)
    color = np.load(_test_data['test'] + 'frame1.npz')['arr_1']
    geo_model = create_example_model(name='Anticline')
    module = GemPyModule(geo_model=geo_model, extent=sensor.extent, box=[1000, 800], load_examples=True,
                         name_example=['Horizontal_layers'])
    module.setup(sensor.get_frame())
    pytest.sb_params['marker'] = aruco.update(frame=color)
    sb_params = module.update(pytest.sb_params)
    sb_params['fig'].show()
    module.set_section_dict((10, 10), (500, 500), "Section1")
    module.set_section_dict((100, 100), (500, 10), "Section2")

    module.set_borehole_dict((500, 500), "borehole3")
    module.set_borehole_dict((900, 500), "borehole4")
    module._get_polygon_data()
    aruco.plot_aruco(sb_params['ax'], sb_params['marker'])
    sb_params['fig'].show()
    widgets = module.show_widgets()
    widgets.show()

