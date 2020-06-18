"""
Module initialisation for sandbox
Created on 15/04/2020

@author: Daniel Escallon
"""
# Main information for all the modules to work (calibration data, projector and sensor)
import os
from .calibration import *

from .projector.projector import Projector

from .sensor import *

# Optional functionality (aruco markers)
from .markers.aruco import ArucoMarkers

# To all the modules to work

from .modules import *

# or all can be sumarized with
from .modules.sandbox_api import *

_test_data = {'topo':  os.path.dirname(
    __file__) + '../notebooks/tutorials/06_LoadSaveTopoModule/saved_DEMs'}

if __name__ == '__main__':
    pass
