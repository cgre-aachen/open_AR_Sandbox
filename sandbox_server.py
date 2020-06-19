# sandbox_server.py
from bokeh.plotting import curdoc
import sandbox as sb
CALLIBRATION_FILE = 'notebooks/calibration_files/my_calibration.json'

calib = sb.CalibrationData(file=CALLIBRATION_FILE)
sensor = sb.Sensor(calib)
projector = sb.Projector(calib, use_panel=True)

aruco = sb.ArucoMarkers(sensor, calib)

module = sb.Sandbox(calib, sensor, projector, aruco)

main_widget = module.show_widgets()

current = main_widget.get_root()
curdoc().add_root(current)
