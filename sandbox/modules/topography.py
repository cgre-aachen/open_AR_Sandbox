from .module_main_thread import Module
from sandbox.markers.aruco import ArucoMarkers


class TopoModule(Module):

    """
    Module for simple Topography visualization without computing a geological model
    """

    # TODO: create widgets
    def __init__(self, *args, **kwargs):
        # call parents' class init, use greyscale colormap as standard and extreme color labeling
        self.height = 2000
        self.fig = None
        self.ax = None

        super().__init__(*args, contours=True,
                         cmap='gist_earth',
                         over='k',
                         under='k',
                         vmin=0,
                         vmax=500,
                         contours_label=True,
                         minor_contours=True,
                         **kwargs)

    def setup(self):
        self.norm = True
        self.plot.minor_contours = True
        frame = self.sensor.get_filtered_frame()
        if self.crop:
            frame = self.crop_frame(frame)
            frame = self.clip_frame(frame)
            frame = self.calib.s_max - frame
        if self.norm: # TODO: include RangeSlider
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

        self.plot.render_frame(frame)


        # if aruco Module is specified:search, update, plot aruco markers
        if isinstance(self.Aruco, ArucoMarkers):
            self.Aruco.search_aruco()
            self.Aruco.update_marker_dict()
            self.Aruco.transform_to_box_coordinates()
            self.plot.plot_aruco(self.Aruco.aruco_markers)

        self.projector.trigger() #triggers the update of the bokeh plot
