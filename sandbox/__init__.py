"""
Module initialisation for sandbox
Created on 15/04/2020

@authors: Daniel Escallon, Simon Virgo, Miguel de la Varga
"""
# Main information for all the modules to work (calibration data, projector and sensor)
import os

_test_data = {'topo': os.path.abspath(os.path.dirname(__file__) +
                                      '/../notebooks/tutorials/06_LoadSaveTopoModule/saved_DEMs/')+os.sep,
              'landslide_topo': os.path.abspath(os.path.dirname(__file__) +
                                                '/../notebooks/tutorials/07_LandslideSimulation/saved_DEMs/')+os.sep,
              'landslide_release': os.path.abspath(os.path.dirname(__file__) +
                                                   '/../notebooks/tutorials/07_LandslideSimulation/'
                                                   'saved_ReleaseAreas/')+os.sep,
              'landslide_simulation': os.path.abspath(os.path.dirname(__file__) +
                                                      '/../notebooks/tutorials/07_LandslideSimulation/'
                                                      'simulation_data/')+os.sep,
              'gempy_data': os.path.abspath(os.path.dirname(__file__) +
                                            '/../notebooks/tutorials/04_GempyModule/Example_Models/inputdata/')+os.sep,
              'test': os.path.abspath(os.path.dirname(__file__) + '/../tests/test_data/')+os.sep,
              'landscape_generation': os.path.abspath(os.path.dirname(__file__) +
                                                      '/../notebooks/tutorials/09_LandscapeGeneration/')+os.sep
              }

_calibration_dir = os.path.abspath(os.path.dirname(__file__) + '/../notebooks/calibration_files/')+os.sep
_package_dir = os.path.dirname(__file__)

# Create folders if not existing
# Topo folder
if not os.path.isdir(_test_data.get("topo")):
    os.mkdir(_test_data.get("topo"))

# Landslides folders
if not os.path.isdir(_test_data.get("landslide_topo")):
    os.mkdir(_test_data.get("landslide_topo"))
if not os.path.isdir(_test_data.get("landslide_release")):
    os.mkdir(_test_data.get("landslide_release"))
if not os.path.isdir(_test_data.get("landslide_simulation")):
    os.mkdir(_test_data.get("landslide_simulation"))

# Test folders
if not os.path.isdir(_test_data.get("test")):
    os.mkdir(_test_data.get("test"))
    os.mkdir(_test_data.get("test")+"temp")

# Landscape folders
if not os.path.isdir(_test_data.get("landscape_generation")+"checkpoints"):
    os.mkdir(_test_data.get("landscape_generation")+"checkpoints")
if not os.path.isdir(_test_data.get("landscape_generation")+"results"):
    os.mkdir(_test_data.get("landscape_generation")+"results")
if not os.path.isdir(_test_data.get("landscape_generation")+"saved_DEMs"):
    os.mkdir(_test_data.get("landscape_generation")+"saved_DEMs")
    os.mkdir(_test_data.get("landscape_generation") + "saved_DEMs/test")


if __name__ == '__main__':
    pass
