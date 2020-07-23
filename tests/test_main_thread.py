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

