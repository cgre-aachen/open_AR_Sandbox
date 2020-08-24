from sandbox import _test_data as test_data
from sandbox.main_thread import MainThread
from sandbox.projector import Projector
from sandbox.sensor import Sensor
import matplotlib.pyplot as plt
import numpy as np
file = np.load(test_data['topo'] + "DEM1.npz")
frame = file['arr_0']
extent = np.asarray([0, frame.shape[1], 0, frame.shape[0], frame.min(), frame.max()])

projector = Projector(use_panel=False)
sensor = Sensor(name='dummy')

def test_init():
    smain = MainThread(sensor, projector)
    print(smain)

def test_update():
    projector_2 = Projector(use_panel=True)
    sensor_2 = Sensor(name='dummy')
    smain = MainThread(sensor_2, projector_2)
    smain.update()

def test_run(): #TODO: runs but does not show contour lines
    projector_2 = Projector(use_panel=True)
    sensor_2 = Sensor(name='dummy')
    smain = MainThread(sensor_2, projector_2)
    smain.run()

def test_thread_functions():
    smain = MainThread(sensor, projector)
    smain.run()
    print('run() working')
    smain.pause()
    print('pause() working')
    smain.resume()
    print('resume() working')
    smain.stop()
    print('stop() working')

def test_thread_kinectv2():
    from sandbox import _calibration_dir
    from sandbox.markers import MarkerDetection
    projector_2 = Projector(use_panel=True)
    sensor_2 = Sensor(name='kinect_v2', calibsensor=_calibration_dir+'my_sensor_calibration.json')
    aruco = MarkerDetection(sensor=sensor_2)
    smain = MainThread(sensor_2, projector_2, aruco)
    smain.sb_params['active_contours'] = True
    smain.sb_params['active_cmap'] = True
    smain.run()

def test_thread_module():
    from sandbox.modules import TopoModule, GradientModule
    proj = Projector(use_panel=True)
    sens = Sensor(name='kinect_v2', invert=True)
    smain = MainThread(sens, proj)

    topo = TopoModule(extent=sens.extent)
    grad = GradientModule(extent=sens.extent)

    smain.modules = [topo]
    smain.run()

def test_bug_no_dpi():
    from sandbox import _calibration_dir
    _calibprojector = _calibration_dir + "my_projector_calibration.json"
    _calibsensor = _calibration_dir + "my_sensor_calibration.json"
    from sandbox.sensor import Sensor
    sensor = Sensor(calibsensor=_calibsensor, name="kinect_v2")
    from sandbox.projector import Projector
    projector = Projector(calibprojector=_calibprojector)
    # Initialize the aruco detection
    from sandbox.markers import MarkerDetection
    aruco = MarkerDetection(sensor=sensor)
    from sandbox.main_thread import MainThread
    main = MainThread(sensor=sensor, projector=projector, aruco=aruco)
    # Start the thread
    main.run()
    #main.ARUCO_ACTIVE = False

def test_bug_no_dpi_no_aruco():
    #import matplotlib.text
    from sandbox import _calibration_dir
    _calibprojector = _calibration_dir + "my_projector_calibration.json"
    _calibsensor = _calibration_dir + "my_sensor_calibration.json"
    from sandbox.sensor import Sensor
    sensor = Sensor(calibsensor=_calibsensor, name="kinect_v2")
    from sandbox.projector import Projector
    projector = Projector(calibprojector=_calibprojector)
    # Initialize the aruco detection
    from sandbox.main_thread import MainThread
    main = MainThread(sensor=sensor, projector=projector)
    # Start the thread
    main.run()
    main.sb_params

def test_with_gempy():
    from sandbox import _calibration_dir
    _calibprojector = _calibration_dir + "my_projector_calibration.json"
    _calibsensor = _calibration_dir + "my_sensor_calibration.json"
    from sandbox.sensor import Sensor
    sensor = Sensor(calibsensor=_calibsensor, name="kinect_v2")
    from sandbox.projector import Projector
    projector = Projector(calibprojector=_calibprojector)
    # Initialize the aruco detection
    from sandbox.main_thread import MainThread
    mainT = MainThread(sensor=sensor, projector=projector)
    # Start the thread
    #mainT.run()
    from sandbox.modules import GemPyModule
    gpsb = GemPyModule(geo_model=None,
                       extent=sensor.extent,
                       box=sensor.physical_dimensions,
                       load_examples=True,
                       name_example=['Fault'])
    mainT.add_module(name='gempy', module=gpsb)
    mainT.update()
