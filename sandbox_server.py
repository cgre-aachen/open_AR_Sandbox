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

from sandbox.sandbox_api import Sandbox
try:
    import gempy
    gempy_module = True
except:
    gempy_module = False
module = Sandbox(sensor=sensor,
                     projector=projector,
                     aruco=aruco,
                     gempy_module=gempy_module)

main_widget = module.show_widgets()

current = main_widget.get_root()
curdoc().add_root(current)
