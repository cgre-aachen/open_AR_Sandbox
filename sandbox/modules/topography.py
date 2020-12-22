import panel as pn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from skimage import measure
from matplotlib.patches import Path, PathPatch

from .template import ModuleTemplate


class TopoModule(ModuleTemplate):
    """
    Module for simple Topography visualization without computing a geological model
    """

    def __init__(self, *args, extent: list = None, **kwargs):
        pn.extension()
        self.max_height = 800
        self.center = 0
        self.min_height = -200
        self.sea = False
        self.sea_contour = False
        self.terrain_cmap = None
        self.create_custom_cmap()

        if extent is not None:
            self.vmin = extent[4]
            self.vmax = extent[5]

        self.normalize = True
        self.norm = None
        self.cmap = None
        self.sea_level_patch = None

        # Settings for sea level polygon
        self.sea_level_polygon_alpha = 0.7
        self.sea_level_polygon_line_thickness = 2.
        self.sea_level_polygon_line_color = "blue"
        self.sea_zorder = 1000

        # for relief shading
        self.relief_shading = True
        self.azdeg = 315
        self.altdeg = 4
        self.ve = 0.25

        return print("TopoModule loaded succesfully")

    def update(self, sb_params: dict):
        frame = sb_params.get('frame')
        extent = sb_params.get('extent')
        ax = sb_params.get('ax')
        frame, extent = self.normalize_topography(frame, extent,
                                                  min_height=self.min_height,
                                                  max_height=self.max_height)
        frame = self.plot(frame, ax)

        sb_params['frame'] = frame
        sb_params['ax'] = ax
        sb_params['cmap'] = self.cmap
        sb_params['norm'] = self.norm
        sb_params['extent'] = extent

        return sb_params

    def plot(self, frame, ax):
        if self.sea:
            self.cmap = self.terrain_cmap
            self.norm = self.set_norm
        else:
            self.cmap = plt.get_cmap("gist_earth")
            self.norm = None

        if self.sea_contour:
            self._delete_path()
            # Add contour polygon of sea level
            path = self.create_paths(frame, self.center)
            self.sea_level_patch = PathPatch(path,
                                             alpha=self.sea_level_polygon_alpha,
                                             linewidth=self.sea_level_polygon_line_thickness,
                                             ec=self.sea_level_polygon_line_color,
                                             zorder=self.sea_zorder)

            plt.pause(0.1)
            # TODO: partial fix for this issue (#3), another workaround is to deactivate the labels from
            # ContourLinesModule
            ax.add_patch(self.sea_level_patch)
        else:
            self._delete_path()

        ls = mcolors.LightSource(azdeg=self.azdeg, altdeg=self.altdeg)
        # cmap = plt.cm.copper
        frame = ls.shade(frame, cmap=self.cmap, vert_exag=self.ve, blend_mode='hsv')

        return frame

    def _delete_path(self):
        """remove sea-level patch, if previously defined"""
        if self.sea_level_patch:
            self.sea_level_patch.remove()
        self.sea_level_patch = None

    def normalize_topography(self, frame, extent, min_height, max_height):
        """
        Change the max an min value of the frame and normalize accordingly
        Args:
            frame: sensor frame
            extent: sensor extent
            max_height: Target max height
            min_height: Target min height

        Returns:
            normalized frame and new extent

        """
        # ToDo: change to object attributes: self.max_height, self.min_height, self.frame, self.extent?
        # first position the frame in 0 if the original extent is not in 0
        if extent[-2] != 0:
            displ = 0 - extent[-2]
            frame = frame - displ
        # calculate how much we need to move the fram eso the 0 value correspond to the approximate 0 in the frame
        # min_height assuming is under 0.

        if min_height < 0:
            displace = min_height * (-1) * (extent[-1] - extent[-2]) / (max_height - min_height)
            frame = frame - displace
            extent[-1] = extent[-1] - displace
            extent[-2] = extent[-2] - displace
            # now we set 2 regions. One above sea level and one below sea level. So now we can normalize these two
            # regions above 0
            frame[frame > 0] = frame[frame > 0] * (max_height / extent[-1])
            # below 0
            frame[frame < 0] = frame[frame < 0] * (min_height / extent[-2])
        elif min_height > 0:
            frame = frame * (max_height - min_height) / (extent[-1] - extent[-2])
            frame = frame + min_height  # just displace all up to start in min_height
        elif min_height == 0:
            frame = frame * max_height / (extent[-1])
        else:
            raise AttributeError
        extent[-1] = max_height  # self.plot.vmax = max_height
        extent[-2] = min_height  # self.plot.vmin = min_height
        return frame, extent

    def create_custom_cmap(self):
        colors_undersea = plt.cm.gist_earth(np.linspace(0, 0.20, 256))
        colors_land = plt.cm.gist_earth(np.linspace(0.35, 1, 256))
        all_colors = np.vstack((colors_undersea, colors_land))
        self.terrain_cmap = mcolors.LinearSegmentedColormap.from_list('terrain_map', all_colors)

    def create_paths(self, frame, contour_val):
        """Create compound path for given contour value

        Args:
            frame: sensor frame
            contour_val (float): value of contour

        Returns:
            path: matplotlib.Path object for contour polygon
        """
        # create padding
        frame_padded = np.pad(frame, pad_width=1, mode='constant', constant_values=np.max(frame) + 1)

        contours = measure.find_contours(frame_padded.T, contour_val)

        # combine values
        contour_comb = np.concatenate(contours, axis=0)

        #
        # generate codes to close polygons
        #

        # First: link all
        codes = [Path.LINETO for _ in range(contour_comb.shape[0])]
        # Next: find ends of each contour and close
        index = 0
        for contour in contours:
            codes[index] = Path.MOVETO
            index += len(contour)
            codes[index - 1] = Path.CLOSEPOLY

        path = Path(contour_comb, codes)  # , codes)

        return path

    @property
    def set_norm(self):
        div_norm = mcolors.TwoSlopeNorm(vmin=self.min_height,
                                        vcenter=self.center,
                                        vmax=self.max_height)
        return div_norm

    def show_widgets(self):
        self._create_widgets()
        panel = pn.Column("### Widgets for Topography normalization",
                          # self._widget_normalize,
                          self._widget_min_height,
                          self._widget_max_height,
                          self._widget_sea,
                          self._widget_sea_contour,
                          self._widget_sea_level)
        return panel

    def _create_widgets(self):
        self._widget_min_height = pn.widgets.Spinner(name="Minimum height of topography", value=self.min_height,
                                                     step=20)
        self._widget_min_height.param.watch(self._callback_min_height, 'value', onlychanged=False)

        self._widget_max_height = pn.widgets.Spinner(name="Maximum height of topography", value=self.max_height,
                                                     step=20)
        self._widget_max_height.param.watch(self._callback_max_height, 'value', onlychanged=False)

        # self._widget_normalize = pn.widgets.Checkbox(name='Normalize maximun and minimun height of topography',
        #                                             value=self.normalize)
        # self._widget_normalize.param.watch(self._callback_normalize, 'value',
        #                               onlychanged=False)

        self._widget_sea_level = pn.widgets.IntSlider(name="Set sea level height",
                                                      start=self.min_height,
                                                      end=self.max_height,
                                                      step=5,
                                                      value=self.center)
        self._widget_sea_level.param.watch(self._callback_sea_level, 'value',
                                           onlychanged=False)

        self._widget_sea = pn.widgets.Checkbox(name='Show sea level',
                                               value=self.sea)
        self._widget_sea.param.watch(self._callback_see, 'value',
                                     onlychanged=False)

        self._widget_sea_contour = pn.widgets.Checkbox(name='Show sea level contour',
                                                       value=self.sea_contour)
        self._widget_sea_contour.param.watch(self._callback_see_contour, 'value',
                                             onlychanged=False)

    def _callback_min_height(self, event):
        self.min_height = event.new
        if self.center < self.min_height:
            self.center = self.min_height + 1
            self._widget_sea_level.value = self.center + 1
        self._widget_sea_level.start = event.new + 1

    def _callback_max_height(self, event):
        self.max_height = event.new
        if self.center > self.max_height:
            self.center = self.max_height - 1
            self._widget_sea_level.value = self.center - 1
        self._widget_sea_level.end = event.new - 1

    # def _callback_normalize(self, event):
    #    self.norm = event.new

    def _callback_sea_level(self, event):
        self.center = event.new

    def _callback_see(self, event):
        self.sea = event.new

    def _callback_see_contour(self, event):
        self.sea_contour = event.new
