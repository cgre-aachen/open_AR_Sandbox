import panel as pn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from skimage import measure
import skimage.transform
from matplotlib.patches import Path, PathPatch
from .template import ModuleTemplate
from sandbox import set_logger, _package_dir
logger = set_logger(__name__)


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
        self.sea_level_patch = None
        self.col = None  # Check if image is loaded
        self.type_fluid = ["water",
                           "lava",
                           "slime"]
        self.name_fluid = self.type_fluid[0]
        self.fluid = None  # Image to create the texture
        self.texture = None  # resultant texture after masking with path
        if extent is not None:
            self.extent = extent
            self.load_fluid()
        self.animate = True
        self._anim = 0  # animation
        # Settings for sea level polygon
        self.path = None
        self.sea_level_polygon_alpha = 0.7
        self.sea_level_polygon_line_thickness = 2.
        self.sea_level_polygon_line_color = mcolors.to_hex("blue")
        self.sea_zorder = 1000
        self.sea_fill = True
        logger.info("TopoModule loaded successfully")

    def update(self, sb_params: dict):
        """
        Acquire the information from th sb_params dict and call the functions to modify
        the frame and/or plot in the axes
        Args:
            sb_params:
        Returns:
        """
        frame = sb_params.get('frame')
        extent = sb_params.get('extent')
        ax = sb_params.get('ax')
        frame, extent = self.normalize_topography(frame, extent,
                                                  min_height=self.min_height,
                                                  max_height=self.max_height)
        self.extent = extent
        self.plot(frame, ax, extent)
        sb_params['frame'] = frame
        sb_params['ax'] = ax
        sb_params['extent'] = extent

        return sb_params

    def plot(self, frame, ax, extent):
        """
        Deals with everything related to plotting in the axes
        Args:
            frame: Sandbox frame
            ax: axes of matplotlib figure to paint on
        Returns:

        """
        if self.sea or self.sea_contour:
            self._delete_path()
            #if self._center != self.center:
            # TODO: Avoid unnecessary calculation of new paths
            try:
                self.path = self.create_paths(frame, self.center)
            except Exception as e:
                logger.error(e, exc_info=True)

            if self.sea:
                self.set_texture(self.path)
                if self.col is None:
                    self.col = ax.imshow(self.texture, origin='lower', aspect='auto', zorder=self.sea_zorder+1,
                                         alpha=self.sea_level_polygon_alpha)
                else:
                    self.col.set_data(self.texture)
                    self.col.set_alpha(self.sea_level_polygon_alpha)
            else:
                self._delete_image()

            if self.sea_contour:
                # Add contour polygon of sea level
                self.sea_level_patch = PathPatch(self.path,
                                                 alpha=self.sea_level_polygon_alpha,
                                                 linewidth=self.sea_level_polygon_line_thickness,
                                                 # ec=self.sea_level_polygon_line_color,
                                                 color=self.sea_level_polygon_line_color,
                                                 zorder=self.sea_zorder,
                                                 fill=self.sea_fill)
                # ContourLinesModule
                ax.add_patch(self.sea_level_patch)
        else:
            self._delete_image()
            self._delete_path()

    def _delete_image(self):
        """Remove sea-texture"""
        if self.col:
            self.col.remove()
        self.col = None

    def _delete_path(self):
        """remove sea-level patch, if previously defined"""
        if self.sea_level_patch:
            self.sea_level_patch.remove()
        self.sea_level_patch = None

    def set_texture(self, path):
        x = np.arange(0, self.extent[1] - 1, 1)
        y = np.arange(0, self.extent[3] - 1, 1)
        xx, yy = np.meshgrid(x, y)
        xy = np.vstack((xx.ravel(), yy.ravel())).T
        mask = path.contains_points(xy)
        present = xy[mask]
        self.texture = self.fluid.copy()
        if self.animate:
            self.animate_texture()
        for i in range(len(present)):
            self.texture[present[i][1], present[i][0], -1] = 1

    def animate_texture(self):
        """
        Move the texture image to give the appearance of moving like waves
        Returns:
        """
        A = self.texture.shape[0] / 5
        w = 2.0 / self.texture.shape[1]
        shift = lambda x: A * np.sin(2.0 * np.pi * (x + self._anim) * w)
        for i in range(self.texture.shape[1]):
            vector = np.roll(np.arange(0, self.texture.shape[0]), int(round(shift(i))))
            self.texture[:, i][:] = self.texture[:, i][:][vector]
        self._anim += 1

    @staticmethod
    def normalize_topography(frame, extent, min_height, max_height):
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
        # first position the frame in 0 if the original extent is not in 0
        if extent[-2] != 0:
            displ = 0 - extent[-2]
            frame = frame - displ
        # calculate how much we need to move the frame so the 0 value correspond to the approximate 0 in the frame
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

    @staticmethod
    def reshape(image, shape: tuple):
        """
        Reshape any image to the desired shape. Change shape of numpy array to desired shape
        Args:
            image:
            shape: sandbox shape
        Returns:
             reshaped frame
        """
        return skimage.transform.resize(image, shape,
                                        order=3, mode='edge', anti_aliasing=True, preserve_range=False)

    def load_fluid(self):
        image = plt.imread(_package_dir+'/modules/img/'+self.name_fluid+'.jpg')
        fluid = self.reshape(image, (self.extent[3], self.extent[1]))  # height, width
        # convert RGB image to RGBA
        self.fluid = np.dstack((fluid, np.zeros((self.extent[3], self.extent[1]))))

    @staticmethod
    def create_paths(frame, contour_val):
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

        # generate codes to close polygons
        # First: link all
        codes = [Path.LINETO for _ in range(contour_comb.shape[0])]
        # Next: find ends of each contour and close
        index = 0
        for contour in contours:
            codes[index] = Path.MOVETO
            index += len(contour)
            codes[index - 1] = Path.CLOSEPOLY
        path = Path(contour_comb, codes)
        return path

    def show_widgets(self):
        self._create_widgets()
        panel = pn.Column("### Widgets for Topography normalization",
                          pn.Row(pn.Column(self._widget_min_height,
                                           self._widget_max_height,
                                           self._widget_sea,
                                           pn.Row(self._widget_sea_contour,
                                                  self._widget_fill),
                                           self._widget_sea_level),
                                 pn.Column(self._widget_animation,
                                           self._widget_type_fluid,
                                           self._widget_transparency,
                                           self._widget_color
                                           )
                                 )
                          )
        return panel

    def _create_widgets(self):
        self._widget_min_height = pn.widgets.Spinner(name="Minimum height of topography", value=self.min_height,
                                                     step=20)
        self._widget_min_height.param.watch(self._callback_min_height, 'value', onlychanged=False)

        self._widget_max_height = pn.widgets.Spinner(name="Maximum height of topography", value=self.max_height,
                                                     step=20)
        self._widget_max_height.param.watch(self._callback_max_height, 'value', onlychanged=False)

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
                                                       value=self.sea_contour, width_policy='min' )
        self._widget_sea_contour.param.watch(self._callback_see_contour, 'value',
                                             onlychanged=False)

        self._widget_fill = pn.widgets.Checkbox(name='Fill contour', value=self.sea_fill, width_policy='min')
        self._widget_fill.param.watch(self._callback_fill, 'value', onlychanged=False)

        self._widget_type_fluid = pn.widgets.Select(name='Select type of texture for the fluid',
                                                    options=self.type_fluid,
                                                    size=3,
                                                    value=self.name_fluid)
        self._widget_type_fluid.param.watch(self._callback_select_fluid, 'value',
                                            onlychanged=False)

        self._widget_transparency = pn.widgets.Spinner(name="Select transparency", value=self.sea_level_polygon_alpha,
                                                       step=0.05)
        self._widget_transparency.param.watch(self._callback_transparency, 'value', onlychanged=False)

        self._widget_color = pn.widgets.ColorPicker(name='Color level contour', value=self.sea_level_polygon_line_color)
        self._widget_color.param.watch(self._callback_color, 'value', onlychanged=False)

        self._widget_animation = pn.widgets.Checkbox(name='Animate wave movement', value=self.animate)
        self._widget_animation.param.watch(self._callback_animation, 'value', onlychanged=False)

    def _callback_fill(self, event): self.sea_fill = event.new

    def _callback_transparency(self, event): self.sea_level_polygon_alpha = event.new

    def _callback_color(self, event): self.sea_level_polygon_line_color = event.new

    def _callback_animation(self, event): self.animate = event.new

    def _callback_select_fluid(self, event):
        self.name_fluid = event.new
        self.load_fluid()

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

    def _callback_sea_level(self, event):
        self.center = event.new

    def _callback_see(self, event):
        self.sea = event.new

    def _callback_see_contour(self, event):
        self.sea_contour = event.new
