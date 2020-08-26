"""
Module initialisation for sandbox
Created on 15/04/2020

@authors: Daniel Escallon, Simon Virgo, Miguel de la Varga
"""
# Main information for all the modules to work (calibration data, projector and sensor)
import os

#from . import projector, sensor, markers, calibration, modules

_test_data = {'topo': os.path.dirname(__file__) +
                       '/../notebooks/tutorials/06_LoadSaveTopoModule/saved_DEMs/',
              'landslide_topo': os.path.dirname(__file__) +
                                '/../notebooks/tutorials/07_LandslideSimulation/saved_DEMs/',
              'landslide_release': os.path.dirname(__file__) +
                                   '/../notebooks/tutorials/07_LandslideSimulation/saved_ReleaseAreas/',
              'landslide_simulation': os.path.dirname(__file__) +
                                   '/../notebooks/tutorials/07_LandslideSimulation/simulation_data/',
              'gempy_data': os.path.dirname(__file__) +
                            '/../notebooks/tutorials/04_GempyModule/Example_Models/inputdata/',
              'test': os.path.dirname(__file__) + '/../tests/test_data/',
              'landscape_generation': os.path.dirname(__file__) +
                                      '/../notebooks/tutorials/09_LandscapeGeneration/'
              }

_calibration_dir = os.path.dirname(__file__) + '/../notebooks/calibration_files/'

#from .main_thread import MainThread
from sandbox.sandbox_api import calibrate_projector, calibrate_sensor, start_server

if __name__ == '__main__':
    pass
