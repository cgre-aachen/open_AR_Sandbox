import panel as pn
pn.extension()
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy

from .template import ModuleTemplate


class TopoModule(ModuleTemplate):

    """
    Module for simple Topography visualization without computing a geological model
    """

    def __init__(self, *args, extent: list = None, **kwargs):
        self.max_height = 2000
        self.center = 500
        self.min_height = 0
        self.see = True
        self.terrain_cmap = None
        self.create_custom_cmap()

        if extent is not None:
            self.vmin = extent[4]
            self.vmax = extent[5]

        self.normalize = True
        self.norm = None
        self.cmap = None

    def update(self, frame, ax, extent):
        if self.normalize:
            frame = extent[-1] - frame
            if self.see:
                self.cmap = self.terrain_cmap
                self.norm = self.set_norm
            else:
                self.cmap = plt.get_cmap("gist_earth")
                self.norm = None
            frame, extent = self.normalize_topography(frame, extent, self.max_height, self.min_height)
        self.plot()

        return frame, extent, ax, self.cmap, self.norm

    def plot(self):
        return None

    def normalize_topography(self, frame, extent, max_height, min_height):
        frame = frame * (max_height / frame.max())
        extent[4] = min_height #self.plot.vmin = min_height
        extent[5] = max_height #self.plot.vmax = max_height

        return frame, extent

    def create_custom_cmap(self):
        colors_undersea = plt.cm.gist_earth(numpy.linspace(0, 0.20, 256))
        colors_land = plt.cm.gist_earth(numpy.linspace(0.35, 1, 256))
        all_colors = numpy.vstack((colors_undersea, colors_land))
        self.terrain_cmap = mcolors.LinearSegmentedColormap.from_list('terrain_map',
                                                                all_colors)
    @property
    def set_norm(self):
        div_norm = mcolors.TwoSlopeNorm(vmin=self.min_height,
                                             vcenter=self.center,
                                             vmax=self.max_height)
        return div_norm

    def widgets(self):
        self._create_widgets()
        panel = pn.Column("### Widgets for Topography normalization",
                          self._widget_normalize,
                          self._widget_max_height,
                          self._widget_see,
                          self._widget_see_level)
        return panel

    def _create_widgets(self):
        self._widget_max_height = pn.widgets.Spinner(name="Maximum height of topography", value= self.max_height, step = 20)
        self._widget_max_height.param.watch(self._callback_max_height, 'value', onlychanged=False)

        self._widget_normalize = pn.widgets.Checkbox(name='Normalize maximun and minimun height of topography',
                                                     value=self.normalize)
        self._widget_normalize.param.watch(self._callback_normalize, 'value',
                                       onlychanged=False)

        self._widget_see_level = pn.widgets.IntSlider(name="Set see level height",
                                                       start=self.min_height,
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
        self._widget_see_level.end = event.new

    def _callback_normalize(self, event):
        self.norm = event.new

    def _callback_see_level(self, event):
        self.center = event.new

    def _callback_see(self, event):
        self.see = event.new




