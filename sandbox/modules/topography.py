from .module_main_thread import Module
import panel as pn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy


class TopoModule(Module):

    """
    Module for simple Topography visualization without computing a geological model
    """

    def __init__(self, *args, **kwargs):
        # call parents' class init, use greyscale colormap as standard and extreme color labeling
        self.max_height = 2000
        self.center = 500
        self.min_height = 0
        self.see = True
        self.terrain_cmap = None
        self.create_custom_cmap()
        self.div_norm = None

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
        frame = self.sensor.get_frame()
        if self.crop:
            frame = self.crop_frame(frame)
            frame = self.clip_frame(frame)
        if self.norm:  # TODO: include RangeSlider
            frame = self.calib.s_max - frame
            self.set_norm()
            if self.see:
                self.plot.cmap = self.terrain_cmap
                self.plot.norm = self.div_norm

            frame = self.normalize_topography(frame, self.max_height, self.center, self.min_height)
        else:
            self.plot.cmap = "gist_earth_r"
            self.plot.vmin = self.calib.s_min
            self.plot.vmax = self.calib.s_max

        self.plot.render_frame(frame)
        self.projector.frame.object = self.plot.figure

    def update(self):
        # with self.lock:
        frame = self.sensor.get_frame()
        if self.crop:
            frame = self.crop_frame(frame)
            frame = self.clip_frame(frame)
        if self.norm:
            frame = self.calib.s_max - frame
            if self.see:
                self.plot.cmap = self.terrain_cmap
                self.plot.norm = self.div_norm

            frame = self.normalize_topography(frame, self.max_height, self.center, self.min_height)
        else:
            self.plot.cmap = "gist_earth_r"
            self.plot.vmin = self.calib.s_min
            self.plot.vmax = self.calib.s_max
            self.plot.norm = None

        self.plot.render_frame(frame)

        # if aruco Module is specified: update, plot aruco markers
        if self.ARUCO_ACTIVE:
            self.update_aruco()
            self.plot.plot_aruco(self.Aruco.aruco_markers)

        self.projector.trigger() #triggers the update of the bokeh plot

    def normalize_topography(self, frame, max_height, center, min_height):
        frame = frame * (max_height / frame.max())
        self.plot.vmin = min_height
        self.plot.vmax = max_height
        return frame

    def create_custom_cmap(self):
        colors_undersea = plt.cm.gist_earth(numpy.linspace(0, 0.20, 256))
        colors_land = plt.cm.gist_earth(numpy.linspace(0.35, 1, 256))
        all_colors = numpy.vstack((colors_undersea, colors_land))
        self.terrain_cmap = mcolors.LinearSegmentedColormap.from_list('terrain_map',
                                                                all_colors)

    def set_norm(self):
        self.div_norm = mcolors.TwoSlopeNorm(vmin=self.min_height,
                                             vcenter=self.center,
                                             vmax=self.max_height)


    def show_widgets(self):
        widget = self.widget_plot_module()
        tabs = pn.Tabs(("Topography", self.widget_topography()),
                       ("Plot", widget))
        return tabs

    def widget_topography(self):
        self._create_widgets()
        panel = pn.Column("### Widgets for Topography normalization",
                          self._widget_normalize,
                          self._widget_max_height,
                          self._widget_see,
                          self._widget_see_level)
        return panel

    def _create_widgets(self):
        self._widget_max_height = pn.widgets.IntSlider(name="Maximum height of topography",
                                                         start=0,
                                                         end=8000,
                                                         value= self.max_height,
                                                         step=10)
        self._widget_max_height.param.watch(self._callback_max_height, 'value',
                                       onlychanged=False)

        self._widget_normalize = pn.widgets.Checkbox(name='Normalize maximun and minimun height of topography',
                                                     value=self.norm)
        self._widget_normalize.param.watch(self._callback_normalize, 'value',
                                       onlychanged=False)

        self._widget_see_level = pn.widgets.IntSlider(name="Set see level height",
                                                       start=0,
                                                       end=self.max_height,
                                                       value=self.center)
        self._widget_see_level.param.watch(self._callback_see_level, 'value',
                                            onlychanged=False)

        self._widget_see = pn.widgets.Checkbox(name='Show see level',
                                                     value=self.see)
        self._widget_see.param.watch(self._callback_see, 'value',
                                           onlychanged=False)

    def _callback_max_height(self, event):
        self.max_height = event.new

    def _callback_normalize(self, event):
        self.norm = event.new

    def _callback_see_level(self, event):
        self.center = event.new
        self.set_norm()

    def _callback_see(self, event):
        self.see = event.new



