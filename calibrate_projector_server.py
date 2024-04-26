# calibrate_sensor_server.py
from bokeh.plotting import curdoc
from sandbox.projector import Projector
from sandbox import _calibration_dir, set_logger
logger = set_logger(__name__)

proj = Projector(use_panel=True, p_width=1920, p_height=1080, show_legend=True, show_colorbar=True)
figure = proj.figure
axes = proj.ax
axes.plot([0,100],[0,100], 'r-*', label="line")
proj.trigger()

widget = proj.calibrate_projector()

current = widget.get_root()
curdoc().add_root(current)