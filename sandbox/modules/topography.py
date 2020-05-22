from .module_main_thread import Module
import panel as pn


class TopoModule(Module):

    """
    Module for simple Topography visualization without computing a geological model
    """

    # TODO: create widgets
    def __init__(self, *args, **kwargs):
        # call parents' class init, use greyscale colormap as standard and extreme color labeling
        self.max_height = 2000
        self.center = None
        self.min_height = 0

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
        if self.norm:  # TODO: include RangeSlider
            frame = self.normalize_topography(frame, self.max_height, self.center, self.min_height)

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
            self.normalize_topography(frame, self.max_height, self.center, self.min_height)

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

    def show_widgets(self):
        tabs = pn.Tabs(("Topography", self.widget_topography()),
                       ("Plot", self.widget_plot_module()))
        return tabs

    def widget_topography(self):
        self._create_widgets()
        widgets = pn.WidgetBox(self._widget_normalize,
                               self._widget_range_slider)

        panel = pn.Column("#Widgets for Topography normalization", widgets)
        return panel

    def _create_widgets(self):
        self._widget_range_slider = pn.widgets.IntRangeSlider(name="Minimum and maximum of topography",
                                                              start=0,
                                                              end=10000,
                                                              value=(self.min_height, self.max_height),
                                                              step=10)
        self._widget_range_slider.param.watch(self._callback_range_slider, 'value',
                                       onlychanged=False)

        self._widget_normalize = pn.widgets.Checkbox(name='Normalize maximun and minimun height of topography',
                                                     value=self.norm)
        self._widget_normalize.param.watch(self._callback_normalize, 'value',
                                       onlychanged=False)

    def _callback_range_slider(self, event):
        self.min_height, self.max_height = event.new

    def _callback_normalize(self, event):
        self.norm = event.new



