# sandbox_server.py
from bokeh.plotting import curdoc
from sandbox import _calibration_dir

_calibprojector = _calibration_dir + "my_projector_calibration.json"
_calibsensor = _calibration_dir + "my_sensor_calibration.json"

from sandbox.projector import Projector
from sandbox.sensor import Sensor
from sandbox.markers import MarkerDetection

projector = Projector(calibprojector=_calibprojector, use_panel=True)
sensor = Sensor(calibsensor=_calibsensor, name="kinect_v2")

external_modules = dict(gempy_module=False,
                        gimli_module=False,
                        torch_module=False,
                        devito_module=False)

from sandbox.sandbox_api import Sandbox
module = Sandbox(sensor=sensor,
                 projector=projector,
                 aruco=None,
                 kwargs_external_modules=external_modules)

main_widget = module.show_widgets()

current = main_widget.get_root()
curdoc().add_root(current)