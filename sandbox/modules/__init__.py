import warnings as warn
from .gradients import GradientModule
from .landslides import LandslideSimulation
from .load_save_topography import LoadSaveTopoModule
from .topography import TopoModule
from .prototyping import PrototypingModule
from .search_methods import SearchMethodsModule
from .block_module.block_module import BlockModule
from .block_module.rms_grid import RMS_Grid
try:
    from .geophysics.geoelectrics import GeoelectricsModule
except:
    warn("Geophysics module will not work. PyGimli dependencies not found")
#try:
#    from .gempy.gempy_module import GemPyModule
#except:
#    warn("Gempy module will not work. Gempy dependencies not found")

if __name__ == '__main__':
    pass