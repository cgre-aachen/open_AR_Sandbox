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
aruco = MarkerDetection(sensor=sensor)

external_modules = dict(gempy_module=True,
                        gimli_module=True,
                        torch_module=True,
                        devito_module=True)

from sandbox.sandbox_api import Sandbox
module = Sandbox(sensor=sensor,
                 projector=projector,
                 aruco=aruco,
                 kwargs_external_modules=external_modules)

main_widget = module.show_widgets()

current = main_widget.get_root()
curdoc().add_root(current)
