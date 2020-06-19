# sandbox_server.py
import sys
sys.path.append('C:/Users/Admin/PycharmProjects/open_AR_Sandbox/')
from bokeh.plotting import curdoc

import sandbox as sb
CALLIBRATION_FILE = 'C:/Users/Admin/PycharmProjects/open_AR_Sandbox/notebooks\calibration_files/my_calibration.json'

#sb.start_server(CALLIBRATION_FILE)

calib = sb.CalibrationData(file=CALLIBRATION_FILE)
sensor = sb.Sensor(calib)
projector = sb.Projector(calib, use_panel=True)

aruco = sb.ArucoMarkers(sensor, calib)

module = sb.Sandbox(calib, sensor, projector, aruco)
#module.start()
main_widget = module.show_widgets()

current = main_widget.get_root()
curdoc().add_root(current)
