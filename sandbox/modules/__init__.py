from .gradients import GradientModule
from .landslides import LandslideSimulation
from .load_save_topography import LoadSaveTopoModule
from .topography import TopoModule
from .prototyping import PrototypingModule
from .gempy.gempy_module import GemPyModule
from .search_methods import SearchMethodsModule
from .block_module.block_module import BlockModule
from .block_module.rms_grid import RMS_Grid
from abc import ABC, abstractmethod
#from .sandbox_api import *


class ModuleTemplate(ABC):
    def __init__(self):

        self.values = None

    @abstractmethod
    def update(self):


        self.plot()
        pass

    @abstractmethod
    def plot(self):
        pass


if __name__ == '__main__':
    pass