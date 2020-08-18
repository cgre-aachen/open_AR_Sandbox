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

