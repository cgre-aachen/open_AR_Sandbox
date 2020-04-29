import traceback
from .module_main_thread import Module
from sandbox.markers.aruco import ArucoMarkers


class PrototypingModule(Module):
    """
    Class for the connectivity between Notebook plotting and sandbox image in live thread
    """
    def __init__(self, *args, **kwargs):
        # call parents' class init, use greyscale colormap as standard and extreme color labeling
        self.height = 2000
        super().__init__(*args, contours=True,
                         cmap='gist_earth',
                         over='k',
                         under='k',
                         vmin=0,
                         vmax=500,
                         contours_label=True,
                         **kwargs)

        self.function_to_run = None
        self.active_connection = False

    def setup(self):
        frame = self.sensor.get_filtered_frame()
        if self.crop:
            self.norm = True
            self.plot.minor_contours = True
            frame = self.crop_frame(frame)
            frame = self.clip_frame(frame)
            frame = self.calib.s_max - frame
        if self.norm:
            frame = frame * (self.height / frame.max())
            self.plot.vmin = 0
            self.plot.vmax = self.height

        self.plot.render_frame(frame)
        self.projector.frame.object = self.plot.figure

    def update(self):
        # with self.lock:
        frame = self.sensor.get_filtered_frame()
        if self.crop:
            frame = self.crop_frame(frame)
            frame = self.clip_frame(frame)
            frame = self.calib.s_max - frame
        if self.norm:
            frame = frame * (self.height / frame.max())
            self.plot.vmin = 0
            self.plot.vmax = self.height

        if self.active_connection:
            self.plot.ax.cla()
            try:
                self.function_to_run()
            except Exception:
                traceback.print_exc()
                self.active_connection = False

        else:
            self.plot.render_frame(frame)


        # if aruco Module is specified:search, update, plot aruco markers
        if isinstance(self.Aruco, ArucoMarkers):
            self.Aruco.search_aruco()
            self.Aruco.update_marker_dict()
            self.Aruco.transform_to_box_coordinates()
            self.plot.plot_aruco(self.Aruco.aruco_markers)

        self.projector.trigger()  # triggers the update of the bokeh plot

    def plot_sandbox(self, func):
        def inner1(*args, **kwargs):
            frame = self.sensor.get_filtered_frame()
            if self.crop:
                frame = self.crop_frame(frame)
                frame = self.clip_frame(frame)
            func(*args, sandbox_ax=self.plot.ax, sandbox_frame=frame, **kwargs)
        return inner1
