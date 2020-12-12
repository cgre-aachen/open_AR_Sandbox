import os
import gempy as gp
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN"
from sandbox import _test_data as test_data
from sandbox.modules.gempy import GemPyModule
from sandbox.modules.gempy.example_models import *
import matplotlib.pyplot as plt
import pytest
import numpy as np

file = np.load(file=test_data['topo'] + "DEM4.npz", allow_pickle=True)#, encoding='bytes', allow_pickle=False)
frame = (file['arr_0'] - np.max(file["arr_0"]))*-1
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
    module.show_boundary = True
    module.show_lith = False
    module.show_hillshades = True
    module.show_contour = True
    module.show_fill_contour = True

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
    geo_model = create_example_model('Anticline')
    module = GemPyModule(geo_model=geo_model, extent=extent, box=[1000, 800], load_examples=False)
    module.setup(frame)
q
    module.set_borehole_dict((10, 20), "borehole1")
    module.set_borehole_dict((200, 500), "borehole2")
    module.set_borehole_dict((500, 500), "borehole3")
    module.set_borehole_dict((500, 900), "borehole4")
    module.set_borehole_dict((100, 100), "borehole5")
    module.set_borehole_dict((600, 700), "borehole6")
    module.set_borehole_dict((200, 150), "borehole7")
    module.set_borehole_dict((150, 200), "borehole8")
    module.set_borehole_dict((50, 800), "borehole9")
    module.set_borehole_dict((100, 500), "borehole10")
    module.set_borehole_dict((700, 200), "borehole11")

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

def test_gempy_imshow_plotting():
    geo_model = create_example_model(name='Anticline')
    gp.compute_model(geo_model)
    gp.plot_2d(geo_model)

def test_complex_model():
    geo_model = gp.create_model('Geological_Model1')
    geo_model = gp.init_data(geo_model, extent=[0, 4000, 0, 2775, 200, 1500], resolution=[100, 10, 100])
    gp.set_interpolator(geo_model, theano_optimizer='fast_run', verbose=[])
    geo_model.add_features(['Fault2', 'Cycle2', 'Fault1', 'Cycle1'])
    geo_model.delete_features('Default series')
    geo_model.add_surfaces(['F2', 'H', 'G', 'F1', 'D', 'C', 'B', 'A'])
    gp.map_stack_to_surfaces(geo_model, {'Fault1': 'F1', 'Fault2': 'F2', 'Cycle2': ['G', 'H']})
    geo_model.set_is_fault(['Fault1', 'Fault2'])

    ###Cycle 1
    # surface B - before F1
    geo_model.add_surface_points(X=584, Y=285, Z=500, surface='B')
    geo_model.add_surface_points(X=494, Y=696, Z=500, surface='B')
    geo_model.add_surface_points(X=197, Y=1898, Z=500, surface='B')
    geo_model.add_surface_points(X=473, Y=2180, Z=400, surface='B')
    geo_model.add_surface_points(X=435, Y=2453, Z=400, surface='B')
    # surface C - before F1
    geo_model.add_surface_points(X=946, Y=188, Z=600, surface='C')
    geo_model.add_surface_points(X=853, Y=661, Z=600, surface='C')
    geo_model.add_surface_points(X=570, Y=1845, Z=600, surface='C')
    geo_model.add_surface_points(X=832, Y=2132, Z=500, surface='C')
    geo_model.add_surface_points(X=767, Y=2495, Z=500, surface='C')
    # Surface D - Before F1
    geo_model.add_surface_points(X=967, Y=1638, Z=800, surface='D')
    geo_model.add_surface_points(X=1095, Y=996, Z=800, surface='D')
    # Adding orientation to Cycle 1
    geo_model.add_orientations(X=832, Y=2132, Z=500, surface='C', orientation=[76, 17.88, 1])
    # surface B - After F1
    geo_model.add_surface_points(X=1447, Y=2554, Z=500, surface='B')
    geo_model.add_surface_points(X=1511, Y=2200, Z=500, surface='B')
    geo_model.add_surface_points(X=1549, Y=629, Z=600, surface='B')
    geo_model.add_surface_points(X=1630, Y=287, Z=600, surface='B')
    # surface C - After F1
    geo_model.add_surface_points(X=1891, Y=2063, Z=600, surface='C')
    geo_model.add_surface_points(X=1605, Y=1846, Z=700, surface='C')
    geo_model.add_surface_points(X=1306, Y=1641, Z=800, surface='C')
    geo_model.add_surface_points(X=1476, Y=979, Z=800, surface='C')
    geo_model.add_surface_points(X=1839, Y=962, Z=700, surface='C')
    geo_model.add_surface_points(X=2185, Y=893, Z=600, surface='C')
    geo_model.add_surface_points(X=2245, Y=547, Z=600, surface='C')
    # Surface D - After F1
    geo_model.add_surface_points(X=2809, Y=2567, Z=600, surface='D')
    geo_model.add_surface_points(X=2843, Y=2448, Z=600, surface='D')
    geo_model.add_surface_points(X=2873, Y=876, Z=700, surface='D')
    # Surface D - After F2
    geo_model.add_surface_points(X=3056, Y=2439, Z=650, surface='D')
    geo_model.add_surface_points(X=3151, Y=1292, Z=700, surface='D')

    ### Fault 1
    # Surface F1
    geo_model.add_surface_points(X=1203, Y=138, Z=600, surface='F1')
    geo_model.add_surface_points(X=1250, Y=1653, Z=800, surface='F1')
    # orientation to Fault 1
    geo_model.add_orientations(X=1280, Y=2525, Z=500, surface='F1', orientation=[272, 90, 1])

    ### Cycle 2
    # Surface G - Before F2
    geo_model.add_surface_points(X=1012, Y=1493, Z=900, surface='G')
    geo_model.add_surface_points(X=1002, Y=1224, Z=900, surface='G')
    geo_model.add_surface_points(X=1579, Y=1376, Z=850, surface='G')
    geo_model.add_surface_points(X=2489, Y=336, Z=750, surface='G')
    geo_model.add_surface_points(X=2814, Y=1848, Z=750, surface='G')
    # Surface H - Before F2
    geo_model.add_surface_points(X=2567, Y=129, Z=850, surface='H')
    geo_model.add_surface_points(X=3012, Y=726, Z=800, surface='H')
    # Orientation to cycle 2
    geo_model.add_orientations(X=1996, Y=47, Z=800, surface='G', orientation=[92, 5.54, 1])
    # Surface G - After F2
    geo_model.add_surface_points(X=3031, Y=2725, Z=800, surface='G')
    geo_model.add_surface_points(X=3281, Y=2314, Z=750, surface='G')
    geo_model.add_surface_points(X=3311, Y=1357, Z=750, surface='G')
    geo_model.add_surface_points(X=3336, Y=898, Z=750, surface='G')
    # Surface H - After F2
    geo_model.add_surface_points(X=3218, Y=1818, Z=890, surface='H')
    geo_model.add_surface_points(X=3934, Y=1207, Z=810, surface='H')
    geo_model.add_surface_points(X=3336, Y=704, Z=850, surface='H')

    ### Fault 2
    geo_model.add_surface_points(X=3232, Y=178, Z=1000, surface='F2')
    geo_model.add_surface_points(X=2912, Y=2653, Z=700, surface='F2')
    # Add orientation to Fault 2
    geo_model.add_orientations(X=3132, Y=951, Z=700, surface='F2', orientation=[85, 90, 1])

    gp.compute_model(geo_model, sort_surfaces=False)

    module = GemPyModule(geo_model = geo_model, extent=extent, box=[1000, 800], load_examples=False)
    module.show_boundary = True
    module.show_lith = True
    module.show_hillshades = True
    module.show_contour = True
    module.show_fill_contour = False

    sb_params = module.update(pytest.sb_params)
    sb_params['fig'].show()