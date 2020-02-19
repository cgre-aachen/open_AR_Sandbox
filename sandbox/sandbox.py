# TODO: Docstring!!!

from abc import ABCMeta
from abc import abstractmethod
import json
import logging
import threading
from time import sleep
from warnings import warn

import matplotlib.pyplot as plt
import matplotlib
import numpy
import pandas as pd
import panel as pn
import pickle
import scipy
from scipy.interpolate import griddata  # for DummySensor
import scipy.ndimage
from scipy.spatial.distance import cdist  # for DummySensor
import skimage  # for resizing of block data
import matplotlib.colors as mcolors

from matplotlib.colors import LightSource

# optional imports
try:
    import freenect  # wrapper for KinectV1
except ImportError:
    print('Freenect module not found, KinectV1 will not work.')

try:
    from pykinect2 import PyKinectV2  # Wrapper for KinectV2 Windows SDK
    from pykinect2 import PyKinectRuntime
except ImportError:
    print('pykinect2 module not found, KinectV2 will not work.')

try:
    import cv2
    from cv2 import aruco
    CV2_IMPORT = True
except ImportError:
    CV2_IMPORT = False
    # warn('opencv is not installed. Object detection will not work')
    pass

try:
    import gempy
    from gempy.core.grid_modules.grid_types import Topography
except ImportError:
    warn('gempy not found, GeoMap Module will not work')


# logging and exception handling
verbose = False
if verbose:
    logging.basicConfig(filename="main.log",
                        filemode='w',
                        level=logging.WARNING,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        )


class Sandbox:
    # Wrapping API-class

    def __init__(self, calibration_file=None, sensor='dummy', projector_resolution=None, **kwargs):
        self.calib = CalibrationData(file=calibration_file)

        if projector_resolution is not None:
            self.calib.p_width = projector_resolution[0]
            self.calib.p_height = projector_resolution[1]

        if sensor == 'kinect1':
            self.sensor = KinectV1(self.calib)
        elif sensor == 'kinect2':
            self.sensor = KinectV2(self.calib)
        else:
            self.sensor = DummySensor(calibrationdata=self.calib)

        self.projector = Projector(self.calib)
        self.module = TopoModule(self.calib, self.sensor, self.projector, **kwargs)
        # self.module = Calibration(self.calib, self.sensor, self.projector, **kwargs)


class CalibrationData(object):
    """
        changes from 0.8alpha to 0.9alpha: introduction of box_width and box_height
        changes from 0.9alpha to 1.0alpha: Introduction of aruco corners position
        changes from 1.0alpha to 1.1alpha: Introduction of aruco pose estimation and camera color intrinsic parameters
    """

    def __init__(self,
                 p_width=1280, p_height=800, p_frame_top=0, p_frame_left=0,
                 p_frame_width=600, p_frame_height=450,
                 s_top=10, s_right=10, s_bottom=10, s_left=10, s_min=700, s_max=1500,
                 box_width=1000.0, box_height=800.0,
                 file=None, aruco_corners=None, camera_mtx=None, camera_dist=None):
        """

        Args:
            p_width:
            p_height:
            p_frame_top:
            p_frame_left:
            p_frame_width:
            p_frame_height:
            s_top:
            s_right:
            s_bottom:
            s_left:
            s_min:
            s_max:
            box_width: physical dimensions of the sandbox along x-axis in millimeters
            box_height: physical dimensions of the sandbox along y-axis in millimeters
            aruco_corners: information of the corners if an aruco marker is used
            file:
        """

        # version identifier (will be changed if new calibration parameters are introduced / removed)
        self.version = "1.1alpha"

        # projector
        self.p_width = p_width
        self.p_height = p_height

        self.p_frame_top = p_frame_top
        self.p_frame_left = p_frame_left
        self.p_frame_width = p_frame_width
        self.p_frame_height = p_frame_height

        # self.p_legend_top =
        # self.p_legend_left =
        # self.p_legend_width =
        # self.p_legend_height =

        # hot area
        # self.p_hot_top =
        # self.p_hot_left =
        # self.p_hot_width =
        # self.p_hot_height =

        # profile area
        # self.p_profile_top =
        # self.p_profile_left =
        # self.p_profile_width =
        # self.p_profile_height =

        # sensor (e.g. Kinect)
        self.s_name = 'generic'  # name to identify the associated sensor device
        self.s_width = 500  # will be updated by sensor init
        self.s_height = 400  # will be updated by sensor init

        self.s_top = s_top
        self.s_right = s_right
        self.s_bottom = s_bottom
        self.s_left = s_left
        self.s_min = s_min
        self.s_max = s_max

        self.box_width = box_width
        self.box_height = box_height

        # Aruco Corners
        self.aruco_corners = aruco_corners
        self.camera_mtx = camera_mtx
        self.camera_dist = camera_dist

        if file is not None:
            self.load_json(file)

    # computed parameters for easy access
    @property
    def s_frame_width(self):
        return self.s_width - self.s_left - self.s_right

    @property
    def s_frame_height(self):
        return self.s_height - self.s_top - self.s_bottom

    @property
    def scale_factor(self):
        return (self.p_frame_width / self.s_frame_width), (self.p_frame_height / self.s_frame_height)

    # JSON import/export
    def load_json(self, file):
        with open(file) as calibration_json:
            data = json.load(calibration_json)
            if data['version'] == self.version:
                self.__dict__ = data
                print("JSON configuration loaded.")
            else:
                print("JSON configuration incompatible.\nPlease recalibrate manually!")

    def save_json(self, file='calibration.json'):
        with open(file, "w") as calibration_json:
            json.dump(self.__dict__, calibration_json)
        print('JSON configuration file saved:', str(file))

    def corners_as_json(self, data):
        x = data.to_json()
        self.aruco_corners = x


class Sensor(object):
    """
    Masterclass for initializing the sensor (e.g. the Kinect).
    Init the kinect and provide a method that returns the scanned depth image as numpy array. Also we do the gaussian
    blurring to get smoother lines.
    """
    __metaclass__ = ABCMeta

    def __init__(self, calibrationdata, filter='gaussian', n_frames=3, sigma_gauss=3):
        self.calib = calibrationdata
        self.calib.s_name = self.name
        self.calib.s_width = self.depth_width
        self.calib.s_height = self.depth_height

        self.id = None
        self.device = None
        self.angle = None

        self.depth = None
        self.color = None
        self.ir_frame_raw = None
        self.ir_frame = None

        # TODO: include filter self.-filter parameters as function defaults
        self.filter = filter  # TODO: deprecate get_filtered_frame, make it switchable in runtime
        self.n_frames = n_frames  # filter parameters
        self.sigma_gauss = sigma_gauss

        self.setup()

    @abstractmethod
    def setup(self):
        # Wildcard: Everything necessary to set up before a frame can be fetched.
        pass

    @abstractmethod
    def get_frame(self):
        # Wildcard: Single fetch operation.
        pass

    def get_filtered_frame(self):
        # collect last n frames in a stack
        depth_array = self.get_frame()
        for i in range(self.n_frames - 1):
            depth_array = numpy.dstack([depth_array, self.get_frame()])
        # calculate mean values ignoring zeros by masking them
        depth_array_masked = numpy.ma.masked_where(depth_array == 0, depth_array)  # needed for V2?
        self.depth = numpy.ma.mean(depth_array_masked, axis=2)
        # apply gaussian filter
        self.depth = scipy.ndimage.filters.gaussian_filter(self.depth, self.sigma_gauss)

        return self.depth


class DummySensor(Sensor):
    name = 'dummy'

    def __init__(self, *args, width=512, height=424, depth_limits=(1170, 1370),
                 corners=True, points_n=4, points_distance=0.3,
                 alteration_strength=0.1, random_seed=None, **kwargs):

        self.depth_width = width
        self.depth_height = height

        self.depth_lim = depth_limits
        self.corners = corners
        self.n = points_n
        # distance in percent of grid diagonal
        self.distance = numpy.sqrt(self.depth_width ** 2 + self.depth_height ** 2) * points_distance
        # alteration_strength: 0 to 1 (maximum 1 equals numpy.pi/2 on depth range)
        self.strength = alteration_strength
        self.seed = random_seed

        self.grid = None
        self.positions = None
        self.os_values = None
        self.values = None

        # call parents' class init
        super().__init__(*args, **kwargs)

    def setup(self):
        # create grid, init values, and init interpolation
        self._create_grid()
        self._pick_positions()
        self._pick_values()
        self._interpolate()
        print("DummySensor initialized.")

    def get_frame(self):
        """

        Returns:

        """
        self._alter_values()
        self._interpolate()
        return self.depth

    def _oscillating_depth(self, random):
        r = (self.depth_lim[1] - self.depth_lim[0]) / 2
        return numpy.sin(random) * r + r + self.depth_lim[0]

    def _create_grid(self):
        # creates 2D grid for given resolution
        x, y = numpy.meshgrid(numpy.arange(0, self.depth_width, 1), numpy.arange(0, self.depth_height, 1))
        self.grid = numpy.stack((x.ravel(), y.ravel())).T
        return True

    def _pick_positions(self):
        '''
        Param:
            grid: Set of possible points to pick from
            n: desired number of points (without corners counting), not guaranteed to be reached
            distance: distance or range between points
        :return:
        '''

        numpy.random.seed(seed=self.seed)
        gl = self.grid.shape[0]
        gw = self.grid.shape[1]
        n = self.n

        if self.corners:
            n += 4
            points = numpy.zeros((n, gw))
            points[1, 0] = self.grid[:, 0].max()
            points[2, 1] = self.grid[:, 1].max()
            points[3, 0] = self.grid[:, 0].max()
            points[3, 1] = self.grid[:, 1].max()
            i = 4  # counter
        else:
            points = numpy.zeros((n, gw))
            # randomly pick initial point
            ipos = numpy.random.randint(0, gl)
            points[0, :2] = self.grid[ipos, :2]
            i = 1  # counter

        while i < n:
            # calculate all distances between remaining candidates and sim points
            dist = cdist(points[:i, :2], self.grid[:, :2])
            # choose candidates which are out of range
            mm = numpy.min(dist, axis=0)
            candidates = self.grid[mm > self.distance]
            # count candidates
            cl = candidates.shape[0]
            if cl < 1:
                break
            # randomly pick candidate and set next point
            pos = numpy.random.randint(0, cl)
            points[i, :2] = candidates[pos, :2]

            i += 1

        # just return valid points if early break occured
        self.positions = points[:i]

        return True

    def _pick_values(self):
        numpy.random.seed(seed=self.seed)
        n = self.positions.shape[0]
        self.os_values = numpy.random.uniform(-numpy.pi, numpy.pi, n)
        self.values = self._oscillating_depth(self.os_values)

    def _alter_values(self):
        # maximum range in both directions the values should be altered
        numpy.random.seed(seed=self.seed)
        os_range = self.strength * (numpy.pi / 2)
        for i, value in enumerate(self.os_values):
            self.os_values[i] = value + numpy.random.uniform(-os_range, os_range)
        self.values = self._oscillating_depth(self.os_values)

    def _interpolate(self):
        inter = griddata(self.positions[:, :2], self.values, self.grid[:, :2], method='cubic', fill_value=0)
        self.depth = inter.reshape(self.depth_height, self.depth_width)


class KinectV1(Sensor):
    # hard coded class attributes for KinectV1's native resolution
    name = 'kinect_v1'
    depth_width = 320
    depth_height = 240
    color_width = 640
    color_height = 480

    # TODO: Check!

    def setup(self):
        warn('Two kernels cannot access the Kinect at the same time. This will lead to a sudden death of the kernel. '
             'Be sure no other kernel is running before you initialize a KinectV1 object.')
        print("looking for kinect...")
        ctx = freenect.init()
        self.device = freenect.open_device(ctx, self.id)
        print(self.id)
        freenect.close_device(self.device)  # TODO Test if this has to be done!
        # get the first Depth frame already (the first one takes much longer than the following)
        self.depth = self.get_frame()
        print("KinectV1 initialized.")

    def set_angle(self, angle):  # TODO: throw out
        """
        Args:
            angle:

        Returns:
            None
        """
        self.angle = angle
        freenect.set_tilt_degs(self.device, self.angle)

    def get_frame(self):
        self.depth = freenect.sync_get_depth(index=self.id, format=freenect.DEPTH_MM)[0]
        self.depth = numpy.fliplr(self.depth)
        return self.depth

    def get_rgb_frame(self):  # TODO: check if this can be thrown out
        """

        Returns:

        """
        self.color = freenect.sync_get_video(index=self.id)[0]
        self.color = numpy.fliplr(self.color)
        return self.color

    def calibrate_frame(self, frame, calibration=None):  # TODO: check if this can be thrown out
        """

        Args:
            frame:
            calibration:

        Returns:

        """
        if calibration is None:
            print("no calibration provided!")
        rotated = scipy.ndimage.rotate(frame, calibration.calibration_data.rot_angle, reshape=False)
        cropped = rotated[calibration.calibration_data.y_lim[0]: calibration.calibration_data.y_lim[1],
                  calibration.calibration_data.x_lim[0]: calibration.calibration_data.x_lim[1]]
        cropped = numpy.flipud(cropped)
        return cropped


class KinectV2(Sensor):
    """
    control class for the KinectV2 based on the Python wrappers of the official Microsoft SDK
    Init the kinect and provides a method that returns the scanned depth image as numpy array.
    Also we do gaussian blurring to get smoother surfaces.

    """

    # hard coded class attributes for KinectV2's native resolution
    name = 'kinect_v2'
    depth_width = 512
    depth_height = 424
    color_width = 1920
    color_height = 1080

    def setup(self):
        self.device = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color |
                                                      PyKinectV2.FrameSourceTypes_Depth |
                                                      PyKinectV2.FrameSourceTypes_Infrared)
        self.depth = self.get_frame()
        self.color = self.get_color()
        # self.ir_frame_raw = self.get_ir_frame_raw()
        # self.ir_frame = self.get_ir_frame()
        print("KinectV2 initialized.")

    def get_frame(self):
        """

        Args:

        Returns:
               2D Array of the shape(424, 512) containing the depth information of the latest frame in mm

        """
        depth_flattened = self.device.get_last_depth_frame()
        self.depth = depth_flattened.reshape(
            (self.depth_height, self.depth_width))  # reshape the array to 2D with native resolution of the kinectV2
        return self.depth

    def get_ir_frame_raw(self):
        """

        Args:

        Returns:
               2D Array of the shape(424, 512) containing the raw infrared intensity in (uint16) of the last frame

        """
        ir_flattened = self.device.get_last_infrared_frame()
        self.ir_frame_raw = numpy.flipud(
            ir_flattened.reshape((self.depth_height,
                                  self.depth_width)))  # reshape the array to 2D with native resolution of the kinectV2
        return self.ir_frame_raw

    def get_ir_frame(self, min=0, max=6000):
        """

        Args:
            min: minimum intensity value mapped to uint8 (will become 0) default: 0
            max: maximum intensity value mapped to uint8 (will become 255) default: 6000
        Returns:
               2D Array of the shape(424, 512) containing the infrared intensity between min and max mapped to uint8 of the last frame

        """
        ir_frame_raw = self.get_ir_frame_raw()
        self.ir_frame = numpy.interp(ir_frame_raw, (min, max), (0, 255)).astype('uint8')
        return self.ir_frame

    def get_color(self):
        color_flattened = self.device.get_last_color_frame()
        resolution_camera = self.color_height * self.color_width  # resolution camera Kinect V2
        # Palette of colors in RGB / Cut of 4th column marked as intensity
        palette = numpy.reshape(numpy.array([color_flattened]), (resolution_camera, 4))[:, [2, 1, 0]]
        position_palette = numpy.reshape(numpy.arange(0, len(palette), 1), (self.color_height, self.color_width))
        self.color = numpy.flipud(palette[position_palette])
        #self.color = palette[position_palette]

        return self.color


class Projector(object):
    dpi = 100  # make sure that figures can be displayed pixel-precise

    css = '''
    body {
      margin:0px;
      background-color: #FFFFFF;
    }
    .panel {
      background-color: #000000;
      overflow: hidden;
    }
    .bk.frame {
    }
    .bk.legend {
      background-color: #16425B;
      color: #CCCCCC;
    }
    .bk.hot {
      background-color: #2896A5;
      color: #CCCCCC;
    }
    .bk.profile {
      background-color: #40C1C7;
      color: #CCCCCC;
    }
    '''

    def __init__(self, calibrationdata, use_panel=True):
        self.calib = calibrationdata

        # flags
        self.enable_legend = False
        self.enable_hot = False
        self.enable_profile = False

        # panel components (panes)
        self.panel = None
        self.frame = None
        self.legend = None
        self.hot = None
        self.profile = None

        if use_panel is True:
            self.create_panel()
            self.start_server()

    def create_panel(self):

        pn.extension(raw_css=[self.css])
        # Create a panel object and serve it within an external bokeh browser that will be opened in a separate window

        # In this special case, a "tight" layout would actually add again white space to the plt canvas,
        # which was already cropped by specifying limits to the axis
        self.frame = pn.pane.Matplotlib(plt.figure(),
                                        width=self.calib.p_frame_width,
                                        height=self.calib.p_frame_height,
                                        margin=[self.calib.p_frame_top, 0, 0, self.calib.p_frame_left],
                                        tight=False,
                                        dpi=self.dpi,
                                        css_classes=['frame']
                                        )
        plt.close()  # close figure to prevent inline display

        if self.enable_legend:
            self.legend = pn.Column("### Legend",
                                    # add parameters from calibration for positioning
                                    width=100,
                                    height=100,
                                    margin=[0, 0, 0, 0],
                                    css_classes=['legend'])

        if self.enable_hot:
            self.hot = pn.Column("### Hot area",
                                 width=100,
                                 height=100,
                                 margin=[0, 0, 0, 0],
                                 css_classes=['hot']
                                 )

        if self.enable_profile:
            self.profile = pn.Column("### Profile",
                                     width=100,
                                     height=100,
                                     margin=[0, 0, 0, 0],
                                     css_classes=['profile']
                                     )

        # Combine panel and deploy bokeh server
        self.sidebar = pn.Column(self.legend, self.hot, self.profile,
                                 margin=[self.calib.p_frame_top, 0, 0, 0],
                                 )

        self.panel = pn.Row(self.frame, self.sidebar,
                            width=self.calib.p_width,
                            height=self.calib.p_height,
                            sizing_mode='fixed',
                            css_classes=['panel']
                            )
        return True

    def start_server(self):
        # TODO: Add specific port? port=4242
        # Check for instances and close them?
        self.panel.show(threaded=False)
        print('Projector initialized and server started.\n'
              'Please position the browser window accordingly and enter fullscreen!')
        return True

    def show(self, figure):
        self.frame.object = figure
        return True

    def trigger(self):
        self.frame.param.trigger('object')
        return True


class Plot:

    dpi = 100  # make sure that figures can be displayed pixel-precise

    def __init__(self, calibrationdata, contours=True, margins=False,
                 vmin=None, vmax=None, cmap=None, over=None, under=None,
                 bad=None, norm=None, lot=None, margin_color='r', margin_alpha=0.5,
                 contours_step=100, contours_width=1.0, contours_color='k',
                 contours_label=False, contours_label_inline=True,
                 contours_label_fontsize=15, contours_label_format='%3.0f', #old args
                 model=None, show_faults=True, show_lith=True, #new args
                 marker_position=None, minor_contours=False, contours_step_minor=50,
                 contours_width_minor = 0.5):
        """Creates a new plot instance.

        Regularly, this creates at least a raster plot (plt.pcolormesh), where contours or margin patches can be added.
        Margin patches are e.g. used for the visualization of sensor calibration margins on an uncropped dataframe.
        The rendered plot is accessible via the 'figure' attribute. Internally only the plot axes will be rendered
        and updated. The dataframe will be supplied via the 'render_frame' method.

        Args:
            calibrationdata (CalibrationData): Instance that contains information of the current calibration values.
            contours (bool): Flag that enables or disables contours plotting.
                (default is True)
            margins (bool): Flag that enables or disables plotting of margin patches.
            vmin (float): ...
            vmax (float): ...
            cmap (str or plt.Colormap): Matplotlib colormap, given as name or instance.
            over (e.g. str): Color used for values above the expected data range.
            under (e.g. str): Color used for values below the expected data range.
            bad (e.g. str): Color used for invalid or masked data values.
            norm: Future feature!
            lot: Future feature!
            margin_color (e.g. str): Color of margin patches if enabled.
            margin_alpha (float): Transparency of margin patches.
                (default is 0.5)
            contours_step (int): Size of step between contour lines in model units.
                (default is 10)
            contours_width (float): Width of contour lines.
                (default is 1.0)
            contours_color (e.g. str): Color of contour lines.
                (default is 'k')
            contours_label (bool): Flag that enables labels on contour lines.
                (default is False)
            contours_label_inline (bool): Partly replace underlying contour line or not.
                (default is True)
            contours_label_fontsize (float or str): Size in points or relative size of contour label.
                (default is 15)
            contours_label_format (string or dict): Format string for the contour label.
                (default is %3.0f)
        """
        self.calib = calibrationdata

        # flags
        self.margins = margins
        self.contours = contours
        self.show_lith = show_lith
        self.show_faults = show_faults
        self.marker_position = marker_position
        self.minor_contours = minor_contours


        # z-range handling
        if vmin is not None:
            self.vmin = vmin
        else:
            self.vmin = self.calib.s_min

        if vmax is not None:
            self.vmax = vmax
        else:
            self.vmax = self.calib.s_max

        self.model = model
        if self.model is not None:
            self.cmap = mcolors.ListedColormap(list(self.model.surfaces.df['color']))
        else:
        # pcolormesh setup
            self.cmap = plt.cm.get_cmap(cmap)
        if over is not None:
            self.cmap.set_over(over, 1.0)
        if under is not None:
            self.cmap.set_under(under, 1.0)
        if bad is not None:
            self.cmap.set_bad(bad, 1.0)
        self.norm = norm  # TODO: Future feature
        self.lot = lot  # TODO: Future feature


        # contours setup
        self.contours_step = contours_step  # levels will be supplied via property function
        self.contours_width = contours_width
        self.contours_color = contours_color
        self.contours_label = contours_label
        self.contours_label_inline = contours_label_inline
        self.contours_label_fontsize = contours_label_fontsize
        self.contours_label_format = contours_label_format

        self.contours_step_minor = contours_step_minor
        self.contours_width_minor = contours_width_minor


        # margin patches setup
        self.margin_color = margin_color
        self.margin_alpha = margin_alpha

        # TODO: save the figure's Matplotlib number to recall?
        # self.number = None
        self.figure = None
        self.ax = None
        self.create_empty_frame()

    def create_empty_frame(self):
        """ Initializes the matplotlib figure and empty axes according to projector calibration.

        The figure can be accessed by its attribute. It will be 'deactivated' to prevent random apperance in notebooks.
        """

        self.figure = plt.figure(figsize=(self.calib.p_frame_width / self.dpi, self.calib.p_frame_height / self.dpi),
                                 dpi=self.dpi)
        self.ax = plt.Axes(self.figure, [0., 0., 1., 1.])
        self.figure.add_axes(self.ax)
        plt.close(self.figure)  # close figure to prevent inline display
        self.ax.set_axis_off()

    def render_frame(self, data, contourdata=None, vmin=None, vmax=None, df_position=None):  # ToDo: use keyword arguments
        """Renders a new frame according to class parameters.

        Resets the plot axes and redraws it with a new data frame, figure object remains untouched.
        If the data frame represents geological information (i.e. not topographical height), an optional data frame
        'contourdata' can be passed.

        Args:
            data (numpy.array): Current data frame representing surface height or geology
            contourdata (numpy.array): Current data frame representing surface height, if data is not height
                (default is None)
        """

        self.ax.cla()  # clear axes to draw new ones on figure
        if vmin is None:
            vmin = self.vmin
        if vmax is None:
            vmax = self.vmax

        self.ax.pcolormesh(data, vmin=vmin, vmax=vmax, cmap=self.cmap, norm=self.norm)

        if self.contours:
            if contourdata is None:
                self.add_contours(data)
            else:
                self.add_contours(contourdata)

        if self.margins:
            self.add_margins()

        if self.marker_position:
            self.add_marker_position(df_position)

    def add_margins(self):
        """ Adds margin patches to the current plot object.
        This is only useful when an uncropped dataframe is passed.
        """

        rec_t = plt.Rectangle((0, self.calib.s_height - self.calib.s_top), self.calib.s_width, self.calib.s_top,
                              fc=self.margin_color, alpha=self.margin_alpha)
        rec_r = plt.Rectangle((self.calib.s_width - self.calib.s_right, 0), self.calib.s_right, self.calib.s_height,
                              fc=self.margin_color, alpha=self.margin_alpha)
        rec_b = plt.Rectangle((0, 0), self.calib.s_width, self.calib.s_bottom,
                              fc=self.margin_color, alpha=self.margin_alpha)
        rec_l = plt.Rectangle((0, 0), self.calib.s_left, self.calib.s_height,
                              fc=self.margin_color, alpha=self.margin_alpha)

        self.ax.add_patch(rec_t)
        self.ax.add_patch(rec_r)
        self.ax.add_patch(rec_b)
        self.ax.add_patch(rec_l)

    @property
    def contours_levels(self):
        """Returns the current contour levels, being aware of changes in calibration."""

        return numpy.arange(self.vmin, self.vmax, self.contours_step)

    @property
    def contours_levels_minor(self):
        """Returns the current contour levels, being aware of changes in calibration."""

        return numpy.arange(self.vmin, self.vmax, self.contours_step_minor)

    def add_contours(self, data, extent=None):
        """Renders contours to the current plot object.
        Uses the different attributes to style contour lines and contour labels.
        """

        contours = self.ax.contour(data,
                                   levels=self.contours_levels,
                                   linewidths=self.contours_width,
                                   colors=self.contours_color,
                                   extent=extent)

        if self.minor_contours:
            self.ax.contour(data,
                            levels=self.contours_levels_minor,
                            linewidths=self.contours_width_minor,
                            colors=self.contours_color,
                            extent=extent)


        if self.contours_label:
            self.ax.clabel(contours,
                           inline=self.contours_label_inline,
                           fontsize=self.contours_label_fontsize,
                           fmt=self.contours_label_format)
                           #extent=extent)

    def update_model(self, model):
        self.model = model
        self.cmap = mcolors.ListedColormap(list(self.model.surfaces.df['color']))

    def add_faults(self):
        self.extract_boundaries(e_faults=True, e_lith=False)

    def add_lith(self):
        self.extract_boundaries(e_faults=False, e_lith=True)

    def extract_boundaries(self, e_faults=False, e_lith=False):
        faults = list(self.model.faults.df[self.model.faults.df['isFault'] == True].index)
        shape = self.model.grid.topography.resolution
        a = self.model.solutions.geological_map[1]
        extent = self.model.grid.topography.extent
        zorder = 2
        counter = a.shape[0]

        if e_faults:
            counters = numpy.arange(0, len(faults), 1)
            c_id = 0  # color id startpoint
        elif e_lith:
            counters = numpy.arange(len(faults), counter, 1)
            c_id = len(faults)  # color id startpoint
        else:
            raise AttributeError

        for f_id in counters:
            block = a[f_id]
            level = self.model.solutions.scalar_field_at_surface_points[f_id][numpy.where(
                self.model.solutions.scalar_field_at_surface_points[f_id] != 0)]

            levels = numpy.insert(level, 0, block.max())
            c_id2 = c_id + len(level)
            if f_id == counters.max():
                levels = numpy.insert(levels, level.shape[0], block.min())
                c_id2 = c_id + len(levels)  # color id endpoint
            block = block.reshape(shape)
            zorder = zorder - (f_id + len(level))

            if f_id >= len(faults):
                self.ax.contourf(numpy.fliplr(block.T), 0, levels=numpy.sort(levels), colors=self.cmap.colors[c_id:c_id2][::-1],
                                 linestyles='solid', origin='lower',
                                 extent=extent, zorder=zorder)
            else:
                self.ax.contour(numpy.fliplr(block.T), 0, levels=numpy.sort(levels), colors=self.cmap.colors[c_id:c_id2][0],
                                linestyles='solid', origin='lower',
                                extent=extent, zorder=zorder)
            c_id += len(level)

    def plot_aruco(self, df_position): # Not implemented yet
        if len(df_position) > 0:

            self.ax.scatter(df_position[df_position['is_inside_box']]['box_x'].values,
                            df_position[df_position['is_inside_box']]['box_y'].values,
                            s=350, facecolors='none', edgecolors='r', linewidths=2)
            for i in range(len(df_position[df_position['is_inside_box']])):
                self.ax.annotate(str(df_position[df_position['is_inside_box']].index[i]),
                                 (df_position[df_position['is_inside_box']]['box_x'].values[i],
                                 df_position[df_position['is_inside_box']]['box_y'].values[i]),
                                 c='r',
                                 fontsize=20,
                                 textcoords='offset pixels',
                                 xytext=(20, 20))
            self.ax.plot(df_position[df_position['is_inside_box']]['box_x'].values,
                         df_position[df_position['is_inside_box']]['box_y'].values, '-r')

            self.ax.set_axis_off()


class Scale(object):
    """
    class that handles the scaling of whatever the sandbox shows and the real world sandbox
    self.extent: 3d extent of the model in the sandbox in model units.
    if no model extent is specified, the physical dimensions of the sandbox (x,y) and the set sensor range (z)
    are used.

    """

    def __init__(self, calibrationdata, xy_isometric=True, extent=None):
        """
        Args:
            calibrationdata:
            xy_isometric:
            extent:
        """
        if isinstance(calibrationdata, CalibrationData):
            self.calibration = calibrationdata
        else:
            raise TypeError("you must pass a valid calibration instance")

        self.xy_isometric = xy_isometric
        self.scale = [None, None, None]
        self.pixel_size = [None, None]
        self.pixel_scale = [None, None]

        if extent is None:  # extent should be array with shape (6,) or convert to list?
            self.extent = numpy.asarray([
                0.0,
                self.calibration.box_width,
                0.0,
                self.calibration.box_height,
                self.calibration.s_min,
                self.calibration.s_max,
            ])

        else:
            self.extent = numpy.asarray(extent)  # check: array with 6 entries!

    @property
    def output_res(self):
        # this is the dimension of the cropped kinect frame
        return self.calibration.s_frame_width, self.calibration.s_frame_height

    def calculate_scales(self):
        """
        calculates the factors for the coordinates transformation kinect-extent

        Returns:
            nothing, but changes in place:
            self.output_res [pixels]: width and height of sandbox image
            self.pixel_scale [modelunits/pixel]: XY scaling factor
            pixel_size [mm/pixel]
            self.scale

        """

        self.pixel_scale[0] = float(self.extent[1] - self.extent[0]) / float(self.output_res[0])
        self.pixel_scale[1] = float(self.extent[3] - self.extent[2]) / float(self.output_res[1])
        self.pixel_size[0] = float(self.calibration.box_width) / float(self.output_res[0])
        self.pixel_size[1] = float(self.calibration.box_height) / float(self.output_res[1])

        # TODO: change the extrent in place!! or create a new extent object that stores the extent after that modification.
        if self.xy_isometric:  # model is extended in one horizontal direction to fit  into box while the scale
            # in both directions is maintained
            print("Aspect ratio of the model is fixed in XY")
            if self.pixel_scale[0] >= self.pixel_scale[1]:
                self.pixel_scale[1] = self.pixel_scale[0]
                print("Model size is limited by X dimension")
            else:
                self.pixel_scale[0] = self.pixel_scale[1]
                print("Model size is limited by Y dimension")

        self.scale[0] = self.pixel_scale[0] / self.pixel_size[0]
        self.scale[1] = self.pixel_scale[1] / self.pixel_size[1]
        self.scale[2] = float(self.extent[5] - self.extent[4]) / (self.calibration.s_max - self.calibration.s_min)
        print("scale in Model units/ mm (X,Y,Z): " + str(self.scale))

    # TODO: manually define zscale and either lower or upper limit of Z, adjust rest accordingly.


class Grid(object):
    """
    class for grid objects. a grid stores the 3D coordinate of each pixel recorded by the kinect in model coordinates
    a calibration object must be provided, it is used to crop the kinect data to the area of interest
    TODO:  The cropping should be done in the kinect class, with calibration_data passed explicitly to the method! Do this for all the cases where calibration data is needed!
    """

    def __init__(self, calibration=None, scale=None, crop_to_resolution=True):
        """

        Args:
            calibration:
            scale:

        Returns:
            None

        """

        self.calibration = calibration
        """
        if isinstance(calibration, Calibration):
            self.calibration = calibration
        else:
            raise TypeError("you must pass a valid calibration instance")
        """
        if isinstance(scale, Scale):
            self.scale = scale
        else:
            self.scale = Scale(calibrationdata=self.calibration)
            print("no scale provided or scale invalid. A default scale instance is used")
        self.depth_grid = None
        self.empty_depth_grid = None
        self.create_empty_depth_grid()
        self.crop = crop_to_resolution

    def create_empty_depth_grid(self):
        """
        Sets up XY grid (Z is empty, that is where the name is coming from)

        Returns:

        """

        """compare:
        grid_list = []
        for x in range(self.output_res[1]):
            for y in range(self.output_res[0]):
                grid_list.append([y * self.scale.pixel_scale[1] + self.scale.extent[2], x * self.scale.pixel_scale[0] + 
                self.scale.extent[0]])
        """

        x_mirrored = numpy.linspace(self.scale.extent[0], self.scale.extent[1], self.scale.output_res[0])
        # above: old way to create the depth grid resulted in a mirrored model! it works but is flipped towards the model in gempy

        x = numpy.linspace(self.scale.extent[1], self.scale.extent[0], self.scale.output_res[0])
        y = numpy.linspace(self.scale.extent[2], self.scale.extent[3], self.scale.output_res[1])
        xx, yy = numpy.meshgrid(x, y, indexing='ij')
        self.empty_depth_grid = numpy.vstack([xx.ravel(), yy.ravel()]).T

        print("the shown extent is [" + str(self.empty_depth_grid[0, 0]) + ", " +
              str(self.empty_depth_grid[-1, 0]) + ", " +
              str(self.empty_depth_grid[0, 1]) + ", " +
              str(self.empty_depth_grid[-1, 1]) + "] "
              )

        # return self.empty_depth_grid

    def update_grid(self, depth): #TODO: Can all this be shortened using the cropped depth image?
        """
        Appends the z (depth) coordinate to the empty depth grid.
        this has to be done every frame while the xy coordinates only change if the calibration or model extent is changed.
        For performance reasons these steps are therefore separated.

        Args:
            depth:

        Returns:

        """

       # filtered_depth = numpy.ma.masked_outside(depth, self.calibration.s_min,
       #                                          self.calibration.s_max)
        filtered_depth = depth
        scaled_depth = self.scale.extent[5] - (
                (filtered_depth - self.calibration.s_min) / (
                self.calibration.s_max -
                self.calibration.s_min) * (self.scale.extent[5] - self.scale.extent[4]))
     #  rotated_depth = scipy.ndimage.rotate(scaled_depth, self.calibration.calibration_data.rot_angle,
     #                                      reshape=False)
        #cropped_depth = rotated_depth[self.calibration.calibration_data.y_lim[0]:
        if self.crop == True:
            cropped_depth = numpy.rot90(scaled_depth[self.calibration.s_bottom:(self.calibration.s_bottom+self.calibration.s_frame_height),
                                         self.calibration.s_left:(self.calibration.s_left+self.calibration.s_frame_width)])
        else:
            cropped_depth = numpy.rot90(scaled_depth)

        #flattened_depth = numpy.reshape(cropped_depth, (numpy.shape(self.empty_depth_grid)[0], 1))
        flattened_depth = cropped_depth.ravel().reshape(self.scale.output_res).ravel()
        depth_grid = numpy.c_[self.empty_depth_grid, flattened_depth]

        self.depth_grid = depth_grid


class Module(object):
    """
    Parent Module with threading methods and abstract attributes and methods for child classes
    """
    __metaclass__ = ABCMeta

    def __init__(self, calibrationdata, sensor, projector, Aruco=None, crop=True, **kwargs):
        self.calib = calibrationdata
        self.sensor = sensor
        self.projector = projector
        self.plot = Plot(self.calib, **kwargs)

        # flags
        self.crop = crop
        self.norm = False # for TopoModule to scale the topography

        # threading
        self.lock = threading.Lock()
        self.thread = None
        self.thread_status = 'stopped'  # status: 'stopped', 'running', 'paused'

        #connect to ArucoMarker class
        if CV2_IMPORT is True:
            self.Aruco = Aruco
        self.automatic_calibration = False
        self.automatic_cropping = False
        # self.setup()

    @abstractmethod
    def setup(self):
        # Wildcard: Everything necessary to set up before a model update can be performed.
        pass

    @abstractmethod
    def update(self):
        # Wildcard: Single model update operation that can be looped in a thread.
        pass

    def thread_loop(self):
        while self.thread_status == 'running':
            self.lock.acquire()
            self.update()
            self.lock.release()

    def run(self):
        if self.thread_status != 'running':
            self.thread_status = 'running'
            self.thread = threading.Thread(target=self.thread_loop, daemon=True, )
            self.thread.start()
            print('Thread started or resumed...')
        else:
            print('Thread already running.')

    def stop(self):
        if self.thread_status is not 'stopped':
            self.thread_status = 'stopped'  # set flag to end thread loop
            self.thread.join()  # wait for the thread to finish
            print('Thread stopped.')
        else:
            print('thread was not running.')

    def pause(self):
        if self.thread_status == 'running':
            self.thread_status = 'paused'  # set flag to end thread loop
            self.thread.join()  # wait for the thread to finish
            print('Thread paused.')
        else:
            print('There is no thread running.')

    def resume(self):
        if self.thread_status != 'stopped':
            self.run()
        else:
            print('Thread already stopped.')

    def depth_mask(self, frame):
        """ Creates a boolean mask with True for all values within the set sensor range and False for every pixel
        above and below. If you also want to use clipping, make sure to use the mask before.
        """

        mask = numpy.ma.getmask(numpy.ma.masked_outside(frame, self.calib.s_min, self.calib.s_max))
        return mask

    def crop_frame(self, frame):
        """ Crops the data frame according to the horizontal margins set up in the calibration
        """

        # TODO: Does not work yet for s_top = 0 and s_right = 0, which currently returns an empty frame!
        # TODO: Workaround: do not allow zeroes in calibration widget and use default value = 1
        # TODO: File numpy issue?
        crop = frame[self.calib.s_bottom:-self.calib.s_top, self.calib.s_left:-self.calib.s_right]
        return crop

    def crop_frame_workaround(self, frame):
        # bullet proof working example
        if self.calib.s_top == 0 and self.calib.s_right == 0:
            crop = frame[self.calib.s_bottom:, self.calib.s_left:]
        elif self.calib.s_top == 0:
            crop = frame[self.calib.s_bottom:, self.calib.s_left:-self.calib.s_right]
        elif self.calib.s_right == 0:
            crop = frame[self.calib.s_bottom:-self.calib.s_top, self.calib.s_left:]
        else:
            crop = frame[self.calib.s_bottom:-self.calib.s_top, self.calib.s_left:-self.calib.s_right]

        return crop

    def clip_frame(self, frame):
        """ Clips all values outside of the sensor range to the set s_min and s_max values.
        If you want to create a mask make sure to call depth_mask before performing the clip.
        """

        clip = numpy.clip(frame, self.calib.s_min, self.calib.s_max)
        return clip


class CalibModule(Module):
    """
    Module for calibration and responsive visualization
    """

    def __init__(self, *args, **kwargs):
        # customization
        self.c_under = '#DBD053'
        self.c_over = '#DB3A34'
        self.c_margin = '#084C61'

        # call parents' class init, use greyscale colormap as standard and extreme color labeling
        super().__init__(*args, contours=True, over=self.c_over, cmap='Greys_r', under=self.c_under, **kwargs)

        self.json_filename = None

        # sensor calibration visualization
        pn.extension()
        self.calib_frame = None  # snapshot of sensor frame, only updated with refresh button
        self.calib_plot = Plot(self.calib, margins=True, contours=True,
                               margin_color=self.c_margin,
                               cmap='Greys_r', over=self.c_over, under=self.c_under)#, **kwargs)
        self.calib_panel_frame = pn.pane.Matplotlib(plt.figure(), tight=False, height=335)
        plt.close()  # close figure to prevent inline display
        self._create_widgets()

    # standard methods
    def setup(self):
        frame = self.sensor.get_filtered_frame()
        if self.crop:
            frame = self.crop_frame(frame)
        self.plot.render_frame(frame)
        self.projector.frame.object = self.plot.figure

        # sensor calibration visualization
        self.calib_frame = self.sensor.get_filtered_frame()
        self.calib_plot.render_frame(self.calib_frame)
        self.calib_panel_frame.object = self.calib_plot.figure

    def update(self):
        frame = self.sensor.get_filtered_frame()
        if self.crop:
            frame = self.crop_frame(frame)
        self.plot.render_frame(frame, vmin=self.calib.s_min, vmax=self.calib.s_max)

        # if aruco Module is specified:search, update, plot aruco markers
        if isinstance(self.Aruco, ArucoMarkers):
            self.Aruco.search_aruco()
            self.Aruco.update_marker_dict()
            self.Aruco.transform_to_box_coordinates()
            self.plot.plot_aruco(self.Aruco.aruco_markers)

        self.projector.trigger()

    def update_calib_plot(self):
        self.calib_plot.render_frame(self.calib_frame)
        self.calib_panel_frame.param.trigger('object')

    # layouts
    def calibrate_projector(self):
        widgets = pn.WidgetBox(self._widget_p_frame_top,
                               self._widget_p_frame_left,
                               self._widget_p_frame_width,
                               self._widget_p_frame_height,
                               self._widget_p_enable_auto_calibration,
                               self._widget_p_automatic_calibration)
        panel = pn.Column("### Projector dashboard arrangement", widgets)
        return panel

    def calibrate_sensor(self):
        widgets = pn.WidgetBox('<b>Distance from edges (pixel)</b>',
                               self._widget_s_top,
                               self._widget_s_right,
                               self._widget_s_bottom,
                               self._widget_s_left,
                               self._widget_s_enable_auto_cropping,
                               self._widget_s_automatic_cropping,
                               pn.layout.VSpacer(height=5),
                               '<b>Distance from sensor (mm)</b>',
                               self._widget_s_min,
                               self._widget_s_max,
                               self._widget_refresh_frame
                               )
        rows = pn.Row(widgets, self.calib_panel_frame)
        panel = pn.Column('### Sensor calibration', rows)
        return panel

    def calibrate_box(self):
        widgets = pn.WidgetBox('<b>Physical dimensions of the sandbox)</b>',
                               self._widget_box_width,
                               self._widget_box_height,
                               )
        panel = pn.Column('### box calibration', widgets)
        return panel

    def calibrate(self):
        tabs = pn.Tabs(('Projector', self.calibrate_projector()),
                       ('Sensor', self.calibrate_sensor()),
                       ('Box Dimensions', self.calibrate_box()),
                       ('Save', pn.WidgetBox(self._widget_json_filename,
                                             self._widget_json_save))
                       )
        return tabs

    def _create_widgets(self):

        # projector widgets and links

        self._widget_p_frame_top = pn.widgets.IntSlider(name='Main frame top margin',
                                                        value=self.calib.p_frame_top,
                                                        start=0,
                                                        end=self.calib.p_height - 20)
        self._widget_p_frame_top.link(self.projector.frame, callbacks={'value': self._callback_p_frame_top})

        self._widget_p_frame_left = pn.widgets.IntSlider(name='Main frame left margin',
                                                         value=self.calib.p_frame_left,
                                                         start=0,
                                                         end=self.calib.p_width - 20)
        self._widget_p_frame_left.link(self.projector.frame, callbacks={'value': self._callback_p_frame_left})

        self._widget_p_frame_width = pn.widgets.IntSlider(name='Main frame width',
                                                          value=self.calib.p_frame_width,
                                                          start=10,
                                                          end=self.calib.p_width)
        self._widget_p_frame_width.link(self.projector.frame, callbacks={'value': self._callback_p_frame_width})

        self._widget_p_frame_height = pn.widgets.IntSlider(name='Main frame height',
                                                           value=self.calib.p_frame_height,
                                                           start=10,
                                                           end=self.calib.p_height)
        self._widget_p_frame_height.link(self.projector.frame, callbacks={'value': self._callback_p_frame_height})

        # Auto- Calibration widgets

        self._widget_p_enable_auto_calibration = pn.widgets.Checkbox(name='Enable Automatic Calibration', value=False)
        self._widget_p_enable_auto_calibration.param.watch(self._callback_enable_auto_calibration, 'value',
                                                           onlychanged=False)

        self._widget_p_automatic_calibration = pn.widgets.Button(name="Run", button_type="success")
        self._widget_p_automatic_calibration.param.watch(self._callback_automatic_calibration, 'clicks',
                                                         onlychanged=False)

        # sensor widgets and links

        self._widget_s_top = pn.widgets.IntSlider(name='Sensor top margin',
                                                  bar_color=self.c_margin,
                                                  value=self.calib.s_top,
                                                  start=1,
                                                  end=self.calib.s_height)
        self._widget_s_top.param.watch(self._callback_s_top, 'value', onlychanged=False)

        self._widget_s_right = pn.widgets.IntSlider(name='Sensor right margin',
                                                    bar_color=self.c_margin,
                                                    value=self.calib.s_right,
                                                    start=1,
                                                    end=self.calib.s_width)
        self._widget_s_right.param.watch(self._callback_s_right, 'value', onlychanged=False)

        self._widget_s_bottom = pn.widgets.IntSlider(name='Sensor bottom margin',
                                                     bar_color=self.c_margin,
                                                     value=self.calib.s_bottom,
                                                     start=1,
                                                     end=self.calib.s_height)
        self._widget_s_bottom.param.watch(self._callback_s_bottom, 'value', onlychanged=False)

        self._widget_s_left = pn.widgets.IntSlider(name='Sensor left margin',
                                                   bar_color=self.c_margin,
                                                   value=self.calib.s_left,
                                                   start=1,
                                                   end=self.calib.s_width)
        self._widget_s_left.param.watch(self._callback_s_left, 'value', onlychanged=False)

        self._widget_s_min = pn.widgets.IntSlider(name='Vertical minimum',
                                                  bar_color=self.c_under,
                                                  value=self.calib.s_min,
                                                  start=0,
                                                  end=2000)
        self._widget_s_min.param.watch(self._callback_s_min, 'value', onlychanged=False)

        self._widget_s_max = pn.widgets.IntSlider(name='Vertical maximum',
                                                  bar_color=self.c_over,
                                                  value=self.calib.s_max,
                                                  start=0,
                                                  end=2000)
        self._widget_s_max.param.watch(self._callback_s_max, 'value', onlychanged=False)

        # Auto cropping widgets:

        self._widget_s_enable_auto_cropping = pn.widgets.Checkbox(name='Enable Automatic Cropping', value=False)
        self._widget_s_enable_auto_cropping.param.watch(self._callback_enable_auto_cropping, 'value',
                                                        onlychanged=False)

        self._widget_s_automatic_cropping = pn.widgets.Button(name="Crop", button_type="success")
        self._widget_s_automatic_cropping.param.watch(self._callback_automatic_cropping, 'clicks',
                                                      onlychanged=False)

        # box widgets:

        # self._widget_s_enable_auto_calibration = CheckboxGroup(labels=["Enable Automatic Sensor Calibration"],
        #                                                                  active=[1])
        self._widget_box_width = pn.widgets.IntSlider(name='width of sandbox in mm',
                                                      bar_color=self.c_margin,
                                                      value=int(self.calib.box_width),
                                                      start=1,
                                                      end=2000)
        self._widget_box_width.param.watch(self._callback_box_width, 'value', onlychanged=False)

        # self._widget_s_automatic_calibration = pn.widgets.Toggle(name="Run", button_type="success")
        self._widget_box_height = pn.widgets.IntSlider(name='height of sandbox in mm',
                                                       bar_color=self.c_margin,
                                                       value=int(self.calib.box_height),
                                                       start=1,
                                                       end=2000)
        self._widget_box_height.param.watch(self._callback_box_height, 'value', onlychanged=False)

        # refresh button

        self._widget_refresh_frame = pn.widgets.Button(name='Refresh sensor frame\n(3 sec. delay)!')
        self._widget_refresh_frame.param.watch(self._callback_refresh_frame, 'clicks', onlychanged=False)

        # save selection

        # Only for reading files --> Is there no location picker in panel widgets???
        # self._widget_json_location = pn.widgets.FileInput(name='JSON location')
        self._widget_json_filename = pn.widgets.TextInput(name='Choose a calibration filename:')
        self._widget_json_filename.param.watch(self._callback_json_filename, 'value', onlychanged=False)
        self._widget_json_filename.value = 'calibration.json'

        self._widget_json_save = pn.widgets.Button(name='Save calibration')
        self._widget_json_save.param.watch(self._callback_json_save, 'clicks', onlychanged=False)

        return True

    # projector callbacks

    def _callback_p_frame_top(self, target, event):
        self.pause()
        # set value in calib
        self.calib.p_frame_top = event.new
        m = target.margin
        n = event.new
        # just changing single indices does not trigger updating of pane
        target.margin = [n, m[1], m[2], m[3]]
        self.resume()

    def _callback_p_frame_left(self, target, event):
        self.pause()
        self.calib.p_frame_left = event.new
        m = target.margin
        n = event.new
        target.margin = [m[0], m[1], m[2], n]
        self.resume()

    def _callback_p_frame_width(self, target, event):
        self.pause()
        self.calib.p_frame_width = event.new
        target.width = event.new
        target.param.trigger('object')
        self.resume()

    def _callback_p_frame_height(self, target, event):
        self.pause()
        self.calib.p_frame_height = event.new
        target.height = event.new
        target.param.trigger('object')
        self.resume()

    # sensor callbacks

    def _callback_s_top(self, event):
        self.pause()
        # set value in calib
        self.calib.s_top = event.new
        # change plot and trigger panel update
        self.update_calib_plot()
        self.resume()

    def _callback_s_right(self, event):
        self.pause()
        self.calib.s_right = event.new
        self.update_calib_plot()
        self.resume()

    def _callback_s_bottom(self, event):
        self.pause()
        self.calib.s_bottom = event.new
        self.update_calib_plot()
        self.resume()

    def _callback_s_left(self, event):
        self.pause()
        self.calib.s_left = event.new
        self.update_calib_plot()
        self.resume()

    def _callback_s_min(self, event):
        self.pause()
        self.calib.s_min = event.new
        self.update_calib_plot()
        self.resume()

    def _callback_s_max(self, event):
        self.pause()
        self.calib.s_max = event.new
        self.update_calib_plot()
        self.resume()

    def _callback_refresh_frame(self, event):
        self.pause()
        sleep(3)
        # only here, get a new frame before updating the plot
        self.calib_frame = self.sensor.get_filtered_frame()
        self.update_calib_plot()
        self.resume()

    def _callback_json_filename(self, event):
        self.json_filename = event.new

    def _callback_json_save(self, event):
        if self.json_filename is not None:
            self.calib.save_json(file=self.json_filename)

    ### box dimensions callbacks:

    def _callback_box_width(self, event):
        self.pause()
        self.calib.box_width = float(event.new)
        # self.update_calib_plot()
        self.resume()

    def _callback_box_height(self, event):
        self.pause()
        self.calib.box_height = float(event.new)
        # self.update_calib_plot()
        self.resume()

    ### Automatic Calibration callback

    def _callback_enable_auto_calibration(self, event):
        self.automatic_calibration = event.new
        if self.automatic_calibration == True:
            self.plot.render_frame(self.Aruco.p_arucoMarker(),  vmin=0, vmax=256)
            self.projector.frame.object = self.plot.figure
        else:
            self.plot.create_empty_frame()
            self.projector.frame.object = self.plot.figure

    def _callback_automatic_calibration(self, event):
        if self.automatic_calibration == True:
            p_frame_left, p_frame_top, p_frame_width, p_frame_height = self.Aruco.move_image()
            self.calib.p_frame_left = p_frame_left
            self.calib.p_frame_top = p_frame_top
            self._widget_p_frame_left.value = self.calib.p_frame_left
            self._widget_p_frame_top.value = self.calib.p_frame_top
            self.calib.p_frame_width = p_frame_width
            self.calib.p_frame_height = p_frame_height
            self._widget_p_frame_width.value = self.calib.p_frame_width
            self._widget_p_frame_height.value = self.calib.p_frame_height
            self.plot.render_frame(self.Aruco.p_arucoMarker(), vmin=0, vmax=256)
            self.projector.frame.object = self.plot.figure
            self.update_calib_plot()


    def _callback_enable_auto_cropping(self, event):
        self.automatic_cropping = event.new

    def _callback_automatic_cropping(self, event):
        if self.automatic_cropping == True:
            self.pause()
            s_top, s_left, s_bottom, s_right = self.Aruco.crop_image_aruco()
            self.calib.s_top = s_top
            self.calib.s_bottom = s_bottom
            self.calib.s_left = s_left
            self.calib.s_right = s_right
            self._widget_s_top.value=self.calib.s_top
            self._widget_s_bottom.value = self.calib.s_bottom
            self._widget_s_left.value = self.calib.s_left
            self._widget_s_right.value = self.calib.s_right
            self.update_calib_plot()
            self.resume()


class TopoModule(Module):

    """
    Module for simple Topography visualization without computing a geological model
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


    def setup(self):
        self.norm = True
        self.plot.minor_contours = True
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


class RMS_Grid():

    def __init__(self):
        """ Class to load RMS grids and convert them to a regular grid to use them in the Block module
        """

        self.nx = None
        self.ny = None
        self.nz = None
        self.block_dict = {}
        self.regular_grid_dict = {}
        # default resolution for the regriding. default is kinect v2 resolution and 100 depth levels
        self.regriding_resolution = [424, 512, 100]
        self.coords_x = None  # arrays to store coordinates of cells
        self.coords_y = None
        self.coords_z = None
        self.data_mask = None  # stores the Livecell information from the VIP  File
        self.reservoir_topography = None
        self.method = 'nearest'
        self.mask_method = 'nearest'

    def load_model_vip(self, infile):
        # parse the file
        f = open(infile, "r")

        while True:  # skip header
            l = f.readline().split()
            if len(l) > 2 and l[1] == "Size":
                break

        # n cells
        l = f.readline().split()
        self.nx = int(l[1])
        self.ny = int(l[2])
        self.nz = int(l[3])
        print('nx ny, nz:')
        print(self.nx, self.ny, self.nz)

        while True:  # skip to coordinates
            l = f.readline().split()
            if len(l) > 0 and l[0] == "CORP":
                print("loading cell positions")
                self.parse_coordinates(f, self.nx, self.ny, self.nz)
                print("coordinates loaded")
                break

        while True:  # skip to Livecell
            l = f.readline().split()
            if len(l) > 0 and l[0] == "LIVECELL":
                self.parse_livecells_vip(f, self.nx, self.ny, self.nz)
                print("Livecells loaded")
                break

        # parse the data
        while True:  # skip to key
            line = f.readline()
            l = line.split()
            if line == '':  # check if the end of file was reached and exit the loop if this is the case
                print('end of file reached')
                break

            elif len(l) >= 2 and l[1] == "VALUE":
                key = l[0]
                try:
                    # parse one block of data and store irt under the given key in the dictionary
                    self.parse_block_vip(f, self.block_dict, key, self.nx, self.ny, self.nz)
                except:
                    print('loading block "' + key + "' failed: not a valid VALUE Format")
                    break

        f.close()  # close the file

    def parse_coordinates(self, current_file, nx, ny, nz):
        f = current_file

        self.coords_x = numpy.empty((nx, ny, nz))
        self.coords_y = numpy.empty((nx, ny, nz))
        self.coords_z = numpy.empty((nx, ny, nz))

        for z in range(nz):

            print('processing coordinates in layer ' + str(z))
            for i in range(3):  # skip Layer(nz)
                f.readline()

            for y in range(ny):
                # print(y)
                for x in range(nx):

                    # skip cell header (each cell)
                    l = f.readline().split()
                    while l[0] == 'C':  # skip header
                        l = f.readline().split()

                    px = []
                    py = []
                    pz = []
                    for i in range(4):
                        # read the corner points
                        px.append(float(l[0]))
                        py.append(float(l[1]))
                        pz.append(float(l[2]))

                        px.append(float(l[3]))
                        py.append(float(l[4]))
                        pz.append(float(l[5]))
                        l = f.readline().split()  # read in next line

                    # calculate the arithmetic mean of all 4 corners elementwise:
                    self.coords_x[x, y, z] = numpy.mean(numpy.array(px))
                    self.coords_y[x, y, z] = numpy.mean(numpy.array(py))
                    self.coords_z[x, y, z] = numpy.mean(numpy.array(pz))

    def parse_livecells_vip(self, current_file, nx, ny, nz):
        data_np = numpy.empty((nx, ny, nz))

        # store pointer position to come back to after the values per line were determined
        pointer = current_file.tell()
        line = current_file.readline().split()
        values_per_line = len(line)
        # print(values_per_line)
        current_file.seek(pointer)  # go back to pointer position

        for z in range(nz):
            for y in range(ny):
                x = 0
                for n in range(nx // values_per_line):  # read values in full lines
                    l = current_file.readline().split()
                    if len(l) < values_per_line:  # if there is an empty line, skip to the next
                        l = current_file.readline().split()
                    for i in range(values_per_line):  # iterate values in the line
                        value = l[i]
                        data_np[x, y, z] = float(value)
                        x = x + 1  # iterate x

                if nx % values_per_line > 0:
                    l = current_file.readline().split()
                    for i in range(nx % values_per_line):  # read values in the last not full line
                        value = l[i]
                        data_np[x, y, z] = float(value)
                        x = x + 1

        self.block_dict['mask'] = data_np

    def parse_block_vip(self, current_file, value_dict, key, nx, ny, nz):
        data_np = numpy.empty((nx, ny, nz))

        f = current_file

        pointer = f.tell()  # store pointer position to come back to after the values per line were determined
        for i in range(3):  # skip header
            f.readline()

        l = f.readline().split()
        values_per_line = len(l)
        # print('values per line: ' + str(values_per_line))
        blocklength = nx // values_per_line
        f.seek(pointer)  # go back to pointer position

        # read block data
        if (nx % values_per_line) != 0:
            blocklength = blocklength + 1

        for z in range(nz):
            for i in range(3):
                l = f.readline().split()
            for y in range(ny):
                x = 0

                for line in range(blocklength):
                    l = f.readline().split()
                    if len(l) < 1:
                        l = f.readline().split()  # skip empty line that occurs if value is dividable by 8
                    while l[0] == "C":
                        l = f.readline().split()  # skip the header lines(can vary from file to file)
                    for i in range(len(l)):
                        try:
                            value = l[i]
                            # data.loc[x,y,z] = value
                            # values.append(value)
                            data_np[x, y, z] = float(value)
                            x = x + 1
                        except:
                            print('failed to parse value ', x, y, z)
                            print(l)
                            x = x + 1
        # print(x, y + 1, z + 1)  # to check if all cells are loaded

        print(key + ' loaded')
        value_dict[key] = data_np

        return True

    def convert_to_regular_grid(self, method=None, mask_method=None):
        # prepare the cell coordinates of the original grid
        x = self.coords_x.ravel()
        y = self.coords_y.ravel()
        z = self.coords_z.ravel()

        # prepare the coordinates of the regular grid cells:
        # define extent:
        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()
        zmin = z.min()
        zmax = z.max()

        # prepare the regular grid:
        gx = numpy.linspace(xmin, xmax, num=self.regriding_resolution[0])
        gy = numpy.linspace(ymin, ymax, num=self.regriding_resolution[1])
        gz = numpy.linspace(zmin, zmax, num=self.regriding_resolution[2])

        a, b, c = numpy.meshgrid(gx, gy, gz)

        grid = numpy.stack((a.ravel(), b.ravel(), c.ravel()), axis=1)

        # iterate over all loaded datasets:
        for key in self.block_dict.keys():
            print("processing grid: ", key)
            if key == 'mask':
                self.block_dict[key][:, :, 0] = 0.0
                self.block_dict[key][0, :,
                :] = 0.0  # exchange outer limits of the box so that nearest neighbour returns zeros outside the box
                self.block_dict[key][-1, :, :] = 0.0
                self.block_dict[key][:, -1, :] = 0.0
                self.block_dict[key][:, 0, :] = 0.0
                self.block_dict[key][:, :, -1] = 0.0
                self.block_dict[key][:, :, 0] = 0.0

            data = self.block_dict[key].ravel()

            if key == 'mask':  # for the mask, fill NaN values with 0.0
                if mask_method == None:
                    mask_method = self.mask_method  # 'linear' or 'nearest'
                data = numpy.nan_to_num(data)  # this does not work with nearest neighbour!

                interp_grid = scipy.interpolate.griddata((x, y, z), data, grid, method=mask_method)

            else:
                if method == None:
                    method = self.method
                interp_grid = scipy.interpolate.griddata((x, y, z), data, grid, method=method)

            # save to dictionary:
            # reshape to originasl dimension BUT WITH X AND Y EXCHANGEND
            self.regular_grid_dict[key] = interp_grid.reshape([self.regriding_resolution[1],
                                                               self.regriding_resolution[0],
                                                               self.regriding_resolution[2]]
                                                              )
            print("done!")

    def create_reservoir_topo(self):
        """
        creates a 2d array with the z values of the reservoir top (the z coordinate of the top layer in the array
        """
        # create 2d grid for lookup:
        x = self.coords_x.ravel()
        y = self.coords_y.ravel()

        # prepare the coordinates of the regular grid cells:
        # define extent:
        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()

        # prepare the regular grid:
        gx = numpy.linspace(xmin, xmax, num=self.regriding_resolution[0])
        gy = numpy.linspace(ymin, ymax, num=self.regriding_resolution[1])
        a, b = numpy.meshgrid(gx, gy)

        grid2d = numpy.stack((a.ravel(), b.ravel()), axis=1)

        top_x = self.coords_x[:, :, 0].ravel()
        top_y = self.coords_y[:, :, 0].ravel()
        top_z = self.coords_z[:, :, 0].ravel()

        topo = scipy.interpolate.griddata((top_x, top_y), top_z, grid2d)  # this has to be done with the linear method!
        self.reservoir_topography = topo.reshape([self.regriding_resolution[1], self.regriding_resolution[0]])

    def save(self, filename):
        """
    saves a list with two entries to a pickle:

        [0] the regridded data blocks in a dictionary
        [1] the reservoir topography map

        """
        pickle.dump([self.regular_grid_dict, self.reservoir_topography], open(filename, "wb"))


class BlockModule(Module):
    # child class of Model

    def __init__(self, calibrationdata, sensor, projector, crop=True, **kwarg):
        super().__init__(calibrationdata, sensor, projector, crop, **kwarg)  # call parent init
        self.block_dict = {}
        self.cmap_dict = {}
        self.displayed_dataset_key = "mask"  # variable to choose displayed dataset in runtime
        self.rescaled_block_dict = {}
        self.reservoir_topography = None
        self.rescaled_reservoir_topography = None
        self.show_reservoir_topo = False
        self.num_contours_reservoir_topo = 10  # number of contours in
        self.reservoir_topography_topo_levels = None  # set in setup and in widget.
        self.result = None  # stores the output array of the current frame

        # #rescaled Version of Livecell information. masking has to be done after scaling because the scaling does not support masked arrays
        # self.rescaled_data_mask = None
        self.index = None  # index to find the cells in the rescaled block modules, corresponding to the topography in the sandbox
        self.widget = None  # widget to change models in runtime
        self.min_sensor_offset = 0
        self.max_sensor_offset = 0
        self.minmax_sensor_offset = 0
        self.original_sensor_min = 0
        self.original_sensor_max = 0
        self.mask_threshold = 0.5  # set the threshold for the mask array, interpolated between 0.0 and 1.0 #obsolete!

        self.num_contour_steps = 20

    def setup(self):
        if self.block_dict is None:
            print("No model loaded. Load a model first with load_module_vip(infile)")
            pass
        elif self.cmap_dict is None:
            self.set_colormaps()
        self.rescale_blocks()
        # self.rescale_mask() #nearest neighbour? obsolete! mask is now part of the block_dict

        self.displayed_dataset_key = list(self.block_dict)[1]

        self.plot.contours_color = 'w'  # Adjust default contour color

        self.projector.frame.object = self.plot.figure  # Link figure to projector

        self.calculate_reservoir_contours()

    def update(self):
        # with self.lock:
        frame = self.sensor.get_filtered_frame()

        if self.crop is True:
            frame = self.crop_frame(frame)
        depth_mask = self.depth_mask(frame)

        ###workaround:resize depth mask
        # depth_mask = skimage.transform.resize(
        #    depth_mask,
        #    (
        #    self.block_dict[self.displayed_dataset_key].shape[0], self.block_dict[self.displayed_dataset_key].shape[1]),
        #    order=0
        # )

        frame = self.clip_frame(frame)

        ##workaround: reshape frame to array size, not the other way around!
        #  frame = skimage.transform.resize(
        #          frame,
        #          (self.block_dict[self.displayed_dataset_key].shape[0], self.block_dict[self.displayed_dataset_key].shape[1]),
        #          order=1
        #  )

        if self.displayed_dataset_key is 'mask':  # check if there is a data_mask, TODO: try except key error
            data = self.rescaled_block_dict[self.displayed_dataset_key]
        #  data = self.block_dict[self.displayed_dataset_key]
        else:  # apply data mask

            data = numpy.ma.masked_where(self.rescaled_block_dict['mask'] < self.mask_threshold,
                                         self.rescaled_block_dict[self.displayed_dataset_key]
                                         )

        zmin = self.calib.s_min
        zmax = self.calib.s_max

        index = (frame - zmin) / (zmax - zmin) * (data.shape[2] - 1.0)  # convert the z dimension to index
        index = index.round()  # round to next integer
        self.index = index.astype('int')

        # querry the array:
        i, j = numpy.indices(data[..., 0].shape)  # create arrays with the indices in x and y
        self.result = data[i, j, self.index]

        self.result = numpy.ma.masked_array(self.result, mask=depth_mask)  # apply the depth mask

        self.plot.ax.cla()

        self.plot.vmin = zmin
        self.plot.vmax = zmax
        cmap = self.cmap_dict[self.displayed_dataset_key][0]
        cmap.set_over('black')
        cmap.set_under('black')
        cmap.set_bad('black')

        norm = self.cmap_dict[self.displayed_dataset_key][1]
        min = self.cmap_dict[self.displayed_dataset_key][2]
        max = self.cmap_dict[self.displayed_dataset_key][3]
        self.plot.cmap = cmap
        self.plot.norm = norm
        self.plot.render_frame(self.result, contourdata=frame, vmin=min, vmax=max)  # plot the current frame

        if self.show_reservoir_topo is True:
            self.plot.ax.contour(self.rescaled_reservoir_topography, levels=self.reservoir_topography_topo_levels)
        # render and display
        # self.plot.ax.axis([0, self.calib.s_frame_width, 0, self.calib.s_frame_height])
        # self.plot.ax.set_axis_off()

        self.projector.trigger()
        # return True

    def load_model(self, model_filename):
        """
        loads a regular grid dataset parsed and prepared with the RMS Grid class.
        the pickled list contains 2 entries:
        1.  The regridded Block dictionary
        2.  a 2d array of the lateral size of the blocks with the z values of the uppermost layer
            (= the shape of the reservoir top surface)
        Args:
            model_filename: string with the path to the file to load

        Returns: nothing, changes in place the

        """
        data_list = pickle.load(open(model_filename, "rb"))
        self.block_dict = data_list[0]
        self.reservoir_topography = data_list[1]
        print('Datasets loaded: ', self.block_dict.keys())

    def create_cmap(self, clist):
        """
        create a matplotlib colormap object from a list of discrete colors
        :param clist: list of colors
        :return: colormap
        """

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('default', clist, N=256)
        return cmap

    def create_norm(self, vmin, vmax):
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        return norm

    def set_colormap(self, key=None, cmap='jet', norm=None):
        min = numpy.nanmin(self.block_dict[key].ravel())  # find min ignoring NaNs
        max = numpy.nanmax(self.block_dict[key].ravel())

        if isinstance(cmap, str):  # get colormap by name
            cmap = matplotlib.cm.get_cmap(name=cmap, lut=None)

        if norm is None:
            norm = self.create_norm(min, max)

        self.cmap_dict[key] = [cmap, norm, min, max]

    def set_colormaps(self, cmap=None, norm=None):
        """
        iterates over all datasets and checks if a colormap has been set. if no colormaps exists it creates one.
        default colormap: jet
        :param cmap:
        :param norm:
        :return:
        """
        for key in self.block_dict.keys():
            if key not in self.cmap_dict.keys():  # add entry if not already in cmap_dict
                self.set_colormap(key)

    def rescale_blocks(self):  # scale the blocks xy Size to the cropped size of the sensor
        for key in self.block_dict.keys():
            rescaled_block = skimage.transform.resize(
                self.block_dict[key],
                (self.calib.s_frame_height, self.calib.s_frame_width),
                order=0
            )

            self.rescaled_block_dict[key] = rescaled_block

        if self.reservoir_topography is not None:  # rescale the topography map
            self.rescaled_reservoir_topography = skimage.transform.resize(
                self.reservoir_topography,
                (self.calib.s_frame_height, self.calib.s_frame_width),
                order=0  # nearest neighbour
            )

    def rescale_mask(self):  # scale the blocks xy Size to the cropped size of the sensor
        rescaled_mask = skimage.transform.resize(
            self.data_mask,
            (self.calib.s_frame_height, self.calib.s_frame_width),
            order=0
        )
        self.rescaled_data_mask = rescaled_mask

    def clear_models(self):
        self.block_dict = {}

    def clear_rescaled_models(self):
        self.rescaled_block_dict = {}

    def clear_cmaps(self):
        self.cmap_dict = {}

    def calculate_reservoir_contours(self):
        min = numpy.nanmin(self.rescaled_reservoir_topography.ravel())
        max = numpy.nanmax(self.rescaled_reservoir_topography.ravel())
        step = (max - min) / float(self.num_contours_reservoir_topo)
        print(min, max, step)
        self.reservoir_topography_topo_levels = numpy.arange(min, max, step=step)

    def widget_mask_threshold(self):
        """
        displays a widget to adjust the mask threshold value

        """
        pn.extension()
        widget = pn.widgets.FloatSlider(name='mask threshold (values smaller than the set threshold will be masked)',
                                        start=0.0, end=1.0, step=0.01, value=self.mask_threshold)

        widget.param.watch(self._callback_mask_threshold, 'value', onlychanged=False)

        return widget

    def _callback_mask_threshold(self, event):
        """
        callback function for the widget to update the self.
        :return:
        """
        # used to be with self.lock:
        self.pause()
        self.mask_threshold = event.new
        self.resume()

    def show_widgets(self):
        self.original_sensor_min = self.calib.s_min  # store original sensor values on start
        self.original_sensor_max = self.calib.s_max

        widgets = pn.WidgetBox(self._widget_model_selector(),
                               self._widget_sensor_top_slider(),
                               self._widget_sensor_bottom_slider(),
                               self._widget_sensor_position_slider(),
                               self._widget_show_reservoir_topography(),
                               self._widget_reservoir_contours_num(),
                               self._widget_contours_num()
                               )

        panel = pn.Column("### Interaction widgets", widgets)
        self.widget = panel
        return panel

    def _widget_model_selector(self):
        """
        displays a widget to toggle between the currently active dataset while the sandbox is running
        Returns:

        """
        pn.extension()
        widget = pn.widgets.RadioButtonGroup(name='Model selector',
                                             options=list(self.block_dict.keys()),
                                             value=self.displayed_dataset_key,
                                             button_type='success')

        widget.param.watch(self._callback_selection, 'value', onlychanged=False)

        return widget

    def _callback_selection(self, event):
        """
        callback function for the widget to update the self.
        :return:
        """
        # used to be with self.lock:
        self.pause()
        self.displayed_dataset_key = event.new
        self.resume()

    def _widget_sensor_top_slider(self):
        """
        displays a widget to toggle between the currently active dataset while the sandbox is running
        Returns:

        """
        pn.extension()
        widget = pn.widgets.IntSlider(name='offset top of the model ', start=-250, end=250, step=1, value=0)

        widget.param.watch(self._callback_top_slider, 'value', onlychanged=False)

        return widget

    def _callback_top_slider(self, event):
        """
        callback function for the widget to update the self.
        :return:
        """
        # used to be with self.lock:
        self.pause()
        self.min_sensor_offset = event.new
        self._update_sensor_calib()
        self.resume()

    def _widget_sensor_bottom_slider(self):
        """
        displays a widget to toggle between the currently active dataset while the sandbox is running
        Returns:

        """
        pn.extension()
        widget = pn.widgets.IntSlider(name='offset bottom of the model ', start=-250, end=250, step=1, value=0)

        widget.param.watch(self._callback_bottom_slider, 'value', onlychanged=False)

        return widget

    def _callback_bottom_slider(self, event):
        """
        callback function for the widget to update the self.
        :return:
        """
        # used to be with self.lock:
        self.pause()
        self.max_sensor_offset = event.new
        self._update_sensor_calib()
        self.resume()

    def _widget_sensor_position_slider(self):
        """
        displays a widget to toggle between the currently active dataset while the sandbox is running
        Returns:

        """
        pn.extension()
        widget = pn.widgets.IntSlider(name='offset the model in vertical direction ', start=-250, end=250, step=1,
                                      value=0)

        widget.param.watch(self._callback_position_slider, 'value', onlychanged=False)

        return widget

    def _callback_position_slider(self, event):
        """
        callback function for the widget to update the self.
        :return:
        """
        # used to be with self.lock:
        self.pause()
        self.minmax_sensor_offset = event.new
        self._update_sensor_calib()
        self.resume()

    def _update_sensor_calib(self):
        self.calib.s_min = self.original_sensor_min + self.min_sensor_offset + self.minmax_sensor_offset
        self.calib.s_max = self.original_sensor_max + self.max_sensor_offset + self.minmax_sensor_offset

    def _widget_show_reservoir_topography(self):
        widget = pn.widgets.Toggle(name='show reservoir top contours',
                                   value=self.show_reservoir_topo,
                                   button_type='success')
        widget.param.watch(self._callback_show_reservoir_topography, 'value', onlychanged=False)

        return widget

    def _callback_show_reservoir_topography(self, event):
        self.pause()
        self.show_reservoir_topo = event.new
        self._update_sensor_calib()
        self.resume()

    def _widget_reservoir_contours_num(self):
        """ Shows a widget that allows to change the contours step size"""

        widget = pn.widgets.IntSlider(name='number of contours in the reservoir topography',
                                      start=0,
                                      end=100,
                                      step=1,
                                      value=round(self.num_contours_reservoir_topo))

        widget.param.watch(self._callback_reservoir_contours_num, 'value', onlychanged=False)
        return widget

    def _callback_reservoir_contours_num(self, event):
        self.pause()
        self.num_contours_reservoir_topo = event.new
        self.calculate_reservoir_contours()
        self.resume()

    def _widget_contours_num(self):
        """ Shows a widget that allows to change the contours step size"""

        widget = pn.widgets.IntSlider(name='number of contours in the sandbox',
                                      start=0,
                                      end=100,
                                      step=1,
                                      value=self.num_contour_steps)

        widget.param.watch(self._callback_contours_num, 'value', onlychanged=False)
        return widget

    def _callback_contours_num(self, event):
        self.pause()
        self.plot.vmin = self.calib.s_min
        self.plot.vmax = self.calib.s_max
        self.num_contour_steps = event.new
        self.plot.contours_step = (self.plot.vmax - self.plot.vmin) / float(self.num_contour_steps)
        self.resume()


class GemPyModule(Module):

    #def __init__(self,  geo_model, calibrationdata, sensor, projector, crop=True, **kwarg):
    def __init__(self,geo_model, *args, ** kwargs):
        super().__init__(*args, **kwargs)  # call parent init


        """

        Args:
            geo_model:
            grid:
            geol_map:
            work_directory:

        Returns:
            None

        """
        # TODO: When we move GeoMapModule import gempy just there


        self.geo_model = geo_model
        self.grid = None
        self.scale = None
        self.plot = None
        self.widget = None
        self.model_dict = None

       # self.fault_line = self.create_fault_line(0, self.geo_model.geo_data_res.n_faults + 0.5001)
       # self.main_contours = self.create_main_contours(self.kinect_grid.scale.extent[4],
       #                                                self.kinect_grid.scale.extent[5])
       # self.sub_contours = self.create_sub_contours(self.kinect_grid.scale.extent[4],
       #                                              self.kinect_grid.scale.extent[5])

       #self.x_grid = range(self.kinect_grid.scale.output_res[0])
       # self.y_grid = range(self.kinect_grid.scale.output_res[1])

        self.plot_topography = True
        self.plot_faults = True

        #dataframe to safe Arucos in model Space:
        self.modelspace_arucos=pd.DataFrame()

    def setup(self):

        self.scale = Scale(self.calib, extent=self.geo_model.grid.regular_grid.extent)        #prepare the scale object
        self.scale.calculate_scales()

        self.grid = Grid(calibration=self.calib, scale=self.scale)
        self.grid.create_empty_depth_grid() # prepare the grid object

        self.init_topography()
       # self.grid.update_grid() #update the grid object for the first time

        self.plot = Plot(self.calib, self.geo_model, vmin=float(self.scale.extent[4]), vmax=float(self.scale.extent[5])) #pass arguments for contours here?

        self.projector.frame.object = self.plot.figure  # Link figure to projector

    def init_topography(self):
        self.grid.update_grid(self.sensor.get_filtered_frame())
        self.geo_model.grid.topography = Topography(self.geo_model.grid.regular_grid)
        self.geo_model.grid.topography.extent = self.scale.extent[:4]
        self.geo_model.grid.topography.resolution = self.scale.output_res
        self.geo_model.grid.topography.values = self.grid.depth_grid
        self.geo_model.grid.topography.values_3D = numpy.dstack([self.grid.depth_grid[:, 0].reshape(self.scale.output_res),
                                                         self.grid.depth_grid[:, 1].reshape(self.scale.output_res),
                                                         self.grid.depth_grid[:, 2].reshape(self.scale.output_res)])
        self.geo_model.grid.set_active('topography')
        self.geo_model.update_from_grid()


    def update(self):
        self.grid.update_grid(self.sensor.get_filtered_frame())
        self.geo_model.grid.topography.values = self.grid.depth_grid
        self.geo_model.grid.topography.values_3D[:, :, 2] = self.grid.depth_grid[:, 2].reshape(
                                                self.geo_model.grid.topography.resolution)
        self.geo_model.grid.update_grid_values()
        self.geo_model.update_from_grid()
        gempy.compute_model(self.geo_model, compute_mesh=False)
        self.plot.update_model(self.geo_model)
        # update the self.plot.figure with new axes

        #prepare the plot object
        self.plot.ax.cla()

        self.plot.add_contours(data=numpy.fliplr(self.geo_model.grid.topography.values_3D[:, :, 2].T),
                               extent=self.geo_model.grid.topography.extent)
        self.plot.add_faults()
        self.plot.add_lith()

        # if aruco Module is specified:search, update, plot aruco markers
        if isinstance(self.Aruco, ArucoMarkers):
            self.Aruco.search_aruco()
            self.Aruco.update_marker_dict()
            self.Aruco.transform_to_box_coordinates()
            self.compute_modelspace_arucos()
            self.plot.plot_aruco(self.modelspace_arucos)

        self.projector.trigger()

        return True

    def compute_modelspace_arucos(self):
        df = self.Aruco.aruco_markers.copy()
        for i in self.Aruco.aruco_markers.index:  # increment counter for not found arucos

            #the combination below works though it should not! Look into scale again!!
            #pixel scale and pixel size should be consistent!
            df.at[i, 'box_x'] = (self.scale.pixel_size[0]*self.Aruco.aruco_markers['box_x'][i])
            df.at[i, 'box_y'] = (self.scale.pixel_scale[1]*self.Aruco.aruco_markers['box_y'][i])
        self.modelspace_arucos = df


    def show_widgets(self, Model_dict):
        self.original_sensor_min = self.calib.s_min  # store original sensor values on start
        self.original_sensor_max = self.calib.s_max

        widgets = pn.WidgetBox(self._widget_model_selector(Model_dict)
                              # self._widget_sensor_top_slider(),
                              # self._widget_sensor_bottom_slider(),
                              # self._widget_sensor_position_slider(),
                              # self._widget_show_reservoir_topography(),
                              # self._widget_reservoir_contours_num(),
                              # self._widget_contours_num()
                               )

        panel = pn.Column("### Interaction widgets", widgets)
        self.widget = panel
        return panel

    def _widget_model_selector(self, Model_dict):
        """
        displays a widget to toggle between the currently active dataset while the sandbox is running
        Returns:

        """
        self.model_dict = Model_dict
        pn.extension()
        widget = pn.widgets.RadioButtonGroup(name='Model selector',
                                             options=list(Model_dict.keys()),
                                             value=list(Model_dict.keys())[0],
                                             button_type='success')

        widget.param.watch(self._callback_selection, 'value', onlychanged=False)

        return widget

    def _callback_selection(self, event):
        """
        callback function for the widget to update the self.
        :return:
        """
        # used to be with self.lock:
        self.stop()
        self.geo_model = self.model_dict[event.new]
        self.setup()
        self.run()


class GeoMapModule:
    """

    """

    def __init__(self, geo_model, grid: Grid, geol_map: Plot):
        """

        Args:
            geo_model:
            grid:
            geol_map:
            work_directory:

        Returns:
            None

        """
        # TODO: When we move GeoMapModule import gempy just there


        self.geo_model = geo_model
        self.kinect_grid = grid
        self.geol_map = geol_map

       # self.fault_line = self.create_fault_line(0, self.geo_model.geo_data_res.n_faults + 0.5001)
       # self.main_contours = self.create_main_contours(self.kinect_grid.scale.extent[4],
       #                                                self.kinect_grid.scale.extent[5])
       # self.sub_contours = self.create_sub_contours(self.kinect_grid.scale.extent[4],
       #                                              self.kinect_grid.scale.extent[5])

        self.x_grid = range(self.kinect_grid.scale.output_res[0])
        self.y_grid = range(self.kinect_grid.scale.output_res[1])

        self.plot_topography = True
        self.plot_faults = True

    def compute_model(self, kinect_array):
        """

        Args:
            kinect_array:

        Returns:

        """
        self.kinect_grid.update_grid(kinect_array)
        sol = gempy.compute_model_at(self.kinect_grid.depth_grid, self.geo_model)
        lith_block = sol[0][0]
        fault_blocks = sol[1][0::2]
        block = lith_block.reshape((self.kinect_grid.scale.output_res[1],
                                    self.kinect_grid.scale.output_res[0]))

        return block, fault_blocks

    # TODO: Miguel: outfile folder should follow by default whatever is set in projection!
    # TODO: Temporal fix. Eventually we need a container class or metaclass with this data
    def render_geo_map(self, block, fault_blocks):
        """

        Args:
            block:
            fault_blocks:

        Returns:

        """

        self.geol_map.render_frame(block)

        elevation = self.kinect_grid.depth_grid.reshape((self.kinect_grid.scale.output_res[1],
                                                         self.kinect_grid.scale.output_res[0], 3))[:, :, 2]
        # This line is for GemPy 1.2: fault_data = sol.fault_blocks.reshape((scalgeol_map.outfilee.output_res[1],
        # scale.output_res[0]))

     #   if self.plot_faults is True:
     #       for fault in fault_blocks:
     #           fault = fault.reshape((self.kinect_grid.scale.output_res[1], self.kinect_grid.scale.output_res[0]))
     #           self.geol_map.add_contours(self.fault_line, [self.x_grid, self.y_grid, fault])
     #   if self.plot_topography is True:
     #       self.geol_map.add_contours(self.main_contours, [self.x_grid, self.y_grid, elevation])
     #       self.geol_map.add_contours(self.sub_contours, [self.x_grid, self.y_grid, elevation])

        return self.geol_map.figure

    def create_fault_line(self,
                          start=0.5,
                          end=50.5,  # TODO Miguel:increase?
                          step=1.0,
                          linewidth=3.0,
                          colors=[(1.0, 1.0, 1.0, 1.0)]):
        """

        Args:
            start:
            end:
            step:
            linewidth:
            colors:

        Returns:

        """

        self.fault_line = Contour(start=start, end=end, step=step, linewidth=linewidth,
                                  colors=colors)

        return self.fault_line

    def create_main_contours(self, start, end, step=100, linewidth=1.0,
                             colors=[(0.0, 0.0, 0.0, 1.0)], show_labels=True):
        """

        Args:
            start:
            end:
            step:
            linewidth:
            colors:
            show_labels:

        Returns:

        """

        self.main_contours = Contour(start=start,
                                     end=end,
                                     step=step,
                                     show_labels=show_labels,
                                     linewidth=linewidth, colors=colors)
        return self.main_contours

    def create_sub_contours(self,
                            start,
                            end,
                            step=25,
                            linewidth=0.8,
                            colors=[(0.0, 0.0, 0.0, 0.8)],
                            show_labels=False
                            ):
        """

        Args:
            start:
            end:
            step:
            linewidth:
            colors:
            show_labels:

        Returns:

        """

        self.sub_contours = Contour(start=start, end=end, step=step, linewidth=linewidth, colors=colors,
                                    show_labels=show_labels)
        return self.sub_contours

    def export_topographic_map(self, output="topographic_map.pdf"):
        """

        Args:
            output:

        Returns:

        """
        self.geol_map.create_empty_frame()
        elevation = self.kinect_grid.depth_grid.reshape((self.kinect_grid.scale.output_res[1],
                                                         self.kinect_grid.scale.output_res[0], 3))[:, :, 2]
        self.geol_map.add_contours(self.main_contours, [self.x_grid, self.y_grid, elevation])
        self.geol_map.add_contours(self.sub_contours, [self.x_grid, self.y_grid, elevation])
        self.geol_map.save(outfile=output)

    def export_geological_map(self, kinect_array, output="geological_map.pdf"):
        """

        Args:
            kinect_array:
            output:

        Returns:

        """

        print("there is still a bug in the map that causes the uppermost lithology to be displayed in the basement"
              " color. Unfortunately we do not have a quick fix for this currently... Sorry! Please fix the map "
              "yourself, for example using illustrator")

        lith_block, fault_blocks = self.compute_model(kinect_array)

        # This line is for GemPy 1.2: lith_block = sol.lith_block.reshape((scale.output_res[1], scale.output_res[0]))

        self.geol_map.create_empty_frame()

        lith_levels = self.geo_model.potential_at_interfaces[-1].sort()
        self.geol_map.add_lith_contours(lith_block, levels=lith_levels)

        elevation = self.kinect_grid.depth_grid.reshape((self.kinect_grid.scale.output_res[1],
                                                         self.kinect_grid.scale.output_res[0], 3))[:, :, 2]
        # This line is for GemPy 1.2: fault_data = sol.fault_blocks.reshape((scalgeol_map.outfilee.output_res[1],
        # scale.output_res[0]))

        if self.plot_faults is True:
            for fault in fault_blocks:
                fault = fault.reshape((self.kinect_grid.scale.output_res[1], self.kinect_grid.scale.output_res[0]))
                self.geol_map.add_contours(self.fault_line, [self.x_grid, self.y_grid, fault])

        if self.plot_topography is True:
            self.geol_map.add_contours(self.main_contours, [self.x_grid, self.y_grid, elevation])
            self.geol_map.add_contours(self.sub_contours, [self.x_grid, self.y_grid, elevation])

        self.geol_map.save(outfile=output)


class GradientModule(Module):
    """
    Module to display the gradient of the topography and the topography as a vector field.
    """

    def __init__(self, *args, **kwargs):
        # call parents' class init, use greyscale colormap as standard and extreme color labeling
        super().__init__(*args, contours=True, cmap='gist_earth_r', over='k', under='k', **kwargs)
        self.frame = None
        self.grad_type = 1

        #lightsource parameter
        self.azdeg = 315
        self.altdeg = 4
        self.ve= 0.25
        self.set_lightsource()

        self.panel_frame = pn.pane.Matplotlib(plt.figure(), tight=False, height=335)
        plt.close()
        self._create_widget_gradients()

    def setup(self):
        self.frame = self.sensor.get_filtered_frame()
        if self.crop:
            self.frame = self.crop_frame(self.frame)
        self.plot.render_frame(self.frame)
        self.projector.frame.object = self.plot.figure

    def set_gradient(self, i):
        self.grad_type = i

    def set_lightsource(self, azdeg=315, altdeg=4, ve=0.25):
        self.azdeg = azdeg
        self.altdeg = altdeg
        self.ve = ve

    def update(self):
        # with self.lock:
        self.frame = self.sensor.get_filtered_frame()
        if self.crop:
            self.frame = self.crop_frame(self.frame) #crops the extent of the kinect image to the sandbox dimensions
            self.frame = self.clip_frame(self.frame) #clips the z values abopve and below the set vertical extent

        self.plot.ax.cla()  # clear axes to draw new ones on figure
        self.plot_grad()

        # if aruco Module is specified:search, update, plot aruco markers
        if isinstance(self.Aruco, ArucoMarkers):
            self.Aruco.search_aruco()
            self.Aruco.update_marker_dict()
            self.Aruco.transform_to_box_coordinates()
            self.plot.plot_aruco(self.Aruco.aruco_markers)

        self.projector.trigger() # trigger update in the projector class

    def plot_grad(self):
        """Create gradient plot and visualize in sandbox"""
        height_map = self.frame
        # self.frame = numpy.clip(height_map, self.frame.min(), 1500.)
        self.frame = numpy.clip(height_map, self.calib.s_min, self.calib.s_max)
        dx, dy = numpy.gradient(self.frame)
        # calculate curvature
        dxdx, dxdy = numpy.gradient(dx)
        dydx, dydy = numpy.gradient(dy)
        laplacian = dxdx + dydy
        #  hillshade
        ls = LightSource(azdeg=self.azdeg, altdeg=self.altdeg)
        rgb = ls.shade(self.frame, cmap=plt.cm.copper, vert_exag=self.ve, blend_mode='hsv')
        #  for quiver
        xx, yy = self.frame.shape

        if self.grad_type == 1:
            self.plot.ax.pcolormesh(dx, cmap='viridis', vmin=-2, vmax=2)
        if self.grad_type == 2:
            self.plot.ax.pcolormesh(dy, cmap='viridis', vmin=-2, vmax=2)
        if self.grad_type == 3:
            self.plot.ax.pcolormesh(numpy.sqrt(dx**2 + dy**2), cmap='viridis', vmin=0, vmax=5)
        if self.grad_type == 4:
            self.plot.ax.pcolormesh(laplacian, cmap='RdBu_r', vmin=-1, vmax=1)
        if self.grad_type == 5:
            self.plot.ax.imshow(rgb, origin='lower left', aspect='auto') # TODO: use pcolormesh insteead of imshow, this method generates axis to the plot
            self.plot.ax.axis('off')
            self.plot.ax.get_xaxis().set_visible(False)
            self.plot.ax.get_yaxis().set_visible(False)
        if self.grad_type == 6:
            self.plot.ax.quiver(numpy.arange(10, yy-10, 10), numpy.arange(10, xx-10, 10),
                                dy[10:-10:10,10:-10:10], dx[10:-10:10,10:-10:10])
        if self.grad_type == 7:
            self.plot.ax.pcolormesh(laplacian, cmap='RdBu_r', vmin=-1, vmax=1)
            self.plot.ax.quiver(numpy.arange(10, yy-10, 10), numpy.arange(10, xx-10, 10),
                                dy[10:-10:10,10:-10:10], dx[10:-10:10,10:-10:10])

        if self.grad_type == 8:
            self.plot.ax.pcolormesh(laplacian, cmap='RdBu_r', vmin=-1, vmax=1)
            self.plot.ax.streamplot(numpy.arange(10, yy-10, 10), numpy.arange(10, xx-10, 10),
                                dy[10:-10:10,10:-10:10], dx[10:-10:10,10:-10:10])

        # streamplot(X, Y, U, V, density=[0.5, 1])

    # Layouts
    def widget_lightsource(self):
        self._widget_azdeg = pn.widgets.FloatSlider(name='Azimuth',

                                                    value=self.azdeg,
                                                    start=0.0,
                                                    end=360.0)
        self._widget_azdeg.param.watch(self._callback_lightsource_azdeg, 'value')

        self._widget_altdeg = pn.widgets.FloatSlider(name='Altitude',
                                                    value=self.altdeg,
                                                    start=0.0,
                                                    end=90.0)
        self._widget_altdeg.param.watch(self._callback_lightsource_altdeg, 'value')

        widgets=pn.WidgetBox('<b>Azimuth</b>',
                             self._widget_azdeg,
                             '<b>Altitude</b>',
                             self._widget_altdeg)

        panel = pn.Column("### Lightsource ", widgets)
        return panel


    def _callback_lightsource_azdeg(self, event):
      #  self.pause()
        self.azdeg = event.new
      #  self.resume()

    def _callback_lightsource_altdeg(self, event):
      #  self.pause()
        self.altdeg = event.new
      #  self.resume()

    def widget_gradients(self):
        widgets = pn.WidgetBox(self._widget_gradient_dx,
                               self._widget_gradient_dy,
                               self._widget_gradient_sqrt,
                               self._widget_laplacian,
                               self._widget_lightsource,
                               self._widget_vector_field,
                               self._widget_laplacian_vector,
                               self._widget_laplacian_stream)

        panel = pn.Column("### Plot gradient model", widgets)
        return panel

    def _create_widget_gradients(self):

        self._widget_gradient_dx = pn.widgets.Button(name = 'Gradient dx', button_type="success")
        self._widget_gradient_dx.param.watch(self._callback_gradient_dx, 'clicks', onlychanged=False)

        self._widget_gradient_dy = pn.widgets.Button(name = 'Gradient dy', button_type="success")
        self._widget_gradient_dy.param.watch(self._callback_gradient_dy, 'clicks', onlychanged=False)

        self._widget_gradient_sqrt = pn.widgets.Button(name = 'Gradient all',button_type="success")
        self._widget_gradient_sqrt.param.watch(self._callback_gradient_sqrt, 'clicks', onlychanged=False)

        self._widget_laplacian = pn.widgets.Button(name = 'Laplacian', button_type="success")
        self._widget_laplacian.param.watch(self._callback_laplacian, 'clicks', onlychanged=False)

        self._widget_lightsource = pn.widgets.Button(name = 'Lightsource', button_type="success")
        self._widget_lightsource.param.watch(self._callback_lightsource, 'clicks', onlychanged=False)

        self._widget_vector_field = pn.widgets.Button(name = 'Vector field', button_type="success")
        self._widget_vector_field.param.watch(self._callback_vector_field, 'clicks', onlychanged=False)

        self._widget_laplacian_vector = pn.widgets.Button(name = 'Laplacian + Vector field', button_type="success")
        self._widget_laplacian_vector.param.watch(self._callback_laplacian_vector, 'clicks', onlychanged=False)

        self._widget_laplacian_stream = pn.widgets.Button(name = 'Laplacian + Stream',button_type="success")
        self._widget_laplacian_stream.param.watch(self._callback_laplacian_stream, 'clicks', onlychanged=False)

        return True

    def _callback_gradient_dx (self, event):
        self.pause()
        self.set_gradient(1)
        self.resume()

    def _callback_gradient_dy (self, event):
        self.pause()
        self.set_gradient(2)
        self.resume()

    def _callback_gradient_sqrt (self, event):
        self.pause()
        self.set_gradient(3)
        self.resume()

    def _callback_laplacian (self, event):
        self.pause()
        self.set_gradient(4)
        self.resume()

    def _callback_lightsource (self, event):
        self.pause()
        self.set_gradient(5)
        self.resume()

    def _callback_vector_field (self, event):
        self.pause()
        self.set_gradient(6)
        self.resume()

    def _callback_laplacian_vector (self, event):
        self.pause()
        self.set_gradient(7)
        self.resume()

    def _callback_laplacian_stream (self, event):
        self.pause()
        self.set_gradient(8)
        self.resume()


class LoadSaveTopoModule(Module):
    """
    Module to save the current topography in a subset of the sandbox
    and recreate it at a later time
    two different representations are saved to the numpy file:

    absolute Topography:
    deviation from the mean height inside the bounding box in millimeter

    relative Height:
    height of each pixel relative to the vmin and vmax of the currently used calibration.
    use relative height with the gempy module to get the same geologic map with different calibration settings.
    """

    def __init__(self, *args, **kwargs):
        # call parents' class init, use greyscale colormap as standard and extreme color labeling
        super().__init__(*args, contours=True, cmap='gist_earth_r', over='k', under='k', **kwargs)
        self.box_origin = (10, 10)  #location of bottom left corner of the box in the sandbox. values refer to pixels of the kinect sensor
        self.box_width = 200
        self.box_height = 150
        self.absolute_topo = None
        self.relative_topo = None

    def setup(self):
        frame = self.sensor.get_filtered_frame()
        if self.crop:
            frame = self.crop_frame(frame)
        self.plot.render_frame(frame)
        self.projector.frame.object = self.plot.figure

    def update(self):
        # with self.lock:
        frame = self.sensor.get_filtered_frame()
        if self.crop:
            frame = self.crop_frame(frame)
        self.plot.render_frame(frame)
        self.showBox(self.box_origin,self.box_width,self.box_height)
        self.projector.trigger()

    def showBox(self, origin, width, height):
        """
        draws a wide rectangle outline in the live view
        :param origin: tuple,relative position from bottom left in sensor pixel space
        :param width: width of box in sensor pixels
        :param height: height of box in sensor pixels

        """
        box = matplotlib.patches.Rectangle(origin, width, height, fill=False, edgecolor='white')
        self.plot.ax.add_patch(box)

    def extractTopo(self):
        frame = self.sensor.get_filtered_frame()
        if self.crop:
            frame = self.crop_frame(frame)

        #crop sensor image to dimensions of box
        cropped_frame = frame[self.box_origin[1]:self.box_origin[1]+self.box_height,
                              self.box_origin[0]:self.box_origin[0]+self.box_width]

        mean_height = cropped_frame.mean()
        self.absolute_topo = cropped_frame-mean_height
        self.relative_topo = self.absolute_topo/(self.calib.s_max-self.calib.s_min)
        return self.absolute_topo, self.relative_topo

    def saveTopo(self, filename="savedTopography.npz"):
        numpy.savez(filename, self.absolute_topo, self.relative_topo)

    def loadTopo(self, filename="savedTopography.npz"):
        files = numpy.load(filename)
        self.absolute_topo = files['arr_0']
        self.relative_topo = files['arr_1']

    def showDifference(self, frame):
        #crop frame
        #create diff
        #crate empty array
        #paste diff array at right location
        #plot
        pass





    def saveTopoVector(self):
        """
        saves a vector graphic of the contour map to disk
        """
        pass


class ArucoMarkers(object):
    """
    class to detect Aruco markers in the kinect data (IR and RGB)
    An Area of interest can be specified, markers outside this area will be ignored
    """

    def __init__(self, sensor, calibration, aruco_dict=None, area=None):
        if not aruco_dict:
            self.aruco_dict = aruco.DICT_4X4_50  # set the default dictionary here
        else:
            self.aruco_dict = aruco_dict
        self.area = area  # set a square Area of interest here (Hot-Area)
        self.kinect = sensor
        self.calib = calibration
        self.ir_markers = None
        if self.calib.aruco_corners is not None:
            self.rgb_markers = pd.read_json(self.calib.aruco_corners)
        else:
            self.rgb_markers = None
        self.projector_markers = None
        self.dict_markers_current = None  # markers that were detected in the last frame
        # self.dict_markers_all = all markers ever detected with their last known position and timestamp
        self.dict_markers_all = self.dict_markers_current
        self.lock = threading.Lock  # thread lock object to avoid read-write collisions in multithreading.
        self.ArucoImage = self.create_aruco_marker()
        self.middle = None
        self.corner_middle = None
        # TODO: correction in x and y direction for the mapping between color space and depth space
        self.correction_x = 8
        self.correction_y = 65
        self.CoordinateMap = self.create_CoordinateMap()

        self.point_markers = None


        #dataframes and variables used in the update loop:
        self.markers_in_frame = pd.DataFrame()
        self.aruco_markers = pd.DataFrame()
        self.threshold = 10.0

        #Pose Estimation
        if self.calib.camera_dist is not None:
            self.mtx = numpy.array((self.calib.camera_mtx))
            self.dist = numpy.array((self.calib.camera_dist))
        else:
            self.mtx = numpy.array([[1977.4905366892494, 0.0, 547.6845435554575], #Hardcoded distorion parameter
                                    [0.0, 2098.757943278828, 962.426967248953],
                                    [0.0, 0.0, 1.0]])
            self.dist = numpy.array([[-0.1521704243263453], #hard-coded distortion parameters
                                     [-0.5137710352422746],
                                     [-0.010673768065933672],
                                     [0.01065954734833698],
                                     [2.2812034123550817],
                                     [0.15820606213404878],
                                     [0.5618247374672848],
                                     [-2.195963638734801],
                                     [0.0],
                                     [0.0],
                                     [0.0],
                                     [0.0],
                                     [0.0],
                                     [0.0]])
        self.size_of_marker = 0.02  # size of aruco markers in meters
        self.length_of_axis = 0.05  # length of the axis drawn on the frame in meters

        #Automatic calibration
        self.load_corners_ids()

    def load_corners_ids(self):
        if self.calib.aruco_corners is not None:
            self.aruco_corners = pd.read_json(self.calib.aruco_corners)
            temp = self.aruco_corners.loc[numpy.argsort(self.aruco_corners.Color_x)[:2]]
            self.corner_id_LU = int(temp.loc[temp.Color_y == temp.Color_y.min()].ids.values)
            temp1 = self.aruco_corners.loc[numpy.argsort(self.aruco_corners.Color_x)[-2:]]
            self.corner_id_DR = int(temp1.loc[temp1.Color_y == temp1.Color_y.max()].ids.values)
            self.center_id = 20

        # TODO: pixel distance from the frame corner so the aruco is always projected inside the sandbox
        self.offset = 100
        # TODO: move the image this amount of pixels so when moving the image is at this distance from the detected aruco
        self.pixel_displacement = 10

    def search_aruco(self):
        """
        searches for aruco markers in the current kinect image and writes detected markers to
         self.markers_in_frame. call this first in the update function.
        :return:
        """

        frame = self.kinect.get_color()
        corners, ids, rejectedImgPoints = self.aruco_detect(frame)
        if ids is not None:
            labels = {"ids", "x", "y", "Counter"}
            df = pd.DataFrame(columns=labels)
            for j in range(len(ids)):
                if ids[j] not in df.ids.values:
                    x_loc, y_loc = self.get_location_marker(corners[j][0])
                    df_temp = pd.DataFrame(
                        {"ids": [ids[j][0]], "x": [x_loc], "y": [y_loc]})
                    df = pd.concat([df, df_temp], sort=False)

            df = df.reset_index(drop=True)
            self.markers_in_frame = self.convert_color_to_depth(None, self.CoordinateMap, data=df)
            self.markers_in_frame.insert(0, 'counter', 0)
            self.markers_in_frame.insert(1, 'box_x', numpy.NaN)
            self.markers_in_frame.insert(2, 'box_y', numpy.NaN)
            self.markers_in_frame.insert(0, 'is_inside_box', numpy.NaN)
            self.markers_in_frame = self.markers_in_frame.set_index(self.markers_in_frame['ids'], drop = True)
            self.markers_in_frame = self.markers_in_frame.drop(columns=['ids'])
        else:
            labels = {"ids", "x", "y", "Counter"}
            self.markers_in_frame = pd.DataFrame(columns=labels)

        return self.markers_in_frame

    def update_marker_dict(self):
        """
        updates existing marker positions in self.aruco_markers. new found markers are auomatically added.
        A marker that is not detected for more than *self.threshold* frames is removed from the list.
        call in update after self.search_aruco():
        :return:
        """
        for j in self.markers_in_frame.index:
            if j not in self.aruco_markers.index:
                self.aruco_markers = self.aruco_markers.append(self.markers_in_frame.loc[j])

            else:
                df_temp = self.markers_in_frame.loc[j]
                self.aruco_markers.at[j] = df_temp

        for i in self.aruco_markers.index:# increment counter for not found arucos
            if i not in self.markers_in_frame.index:
                self.aruco_markers.at[i, 'counter'] += 1.0

            if self.aruco_markers.loc[i]['counter'] >= self.threshold:
                self.aruco_markers = self.aruco_markers.drop(i)

        #return self.aruco_markers

    def transform_to_box_coordinates(self):
        """
        checks if aruco markers are within the dimensions of the sandbox (boolean: is_inside_box)
        and converts the location to box coordinates x,y. call after self.update_markers in the update loop
        :return:
        """
        if len(self.aruco_markers)>0:
            self.aruco_markers['box_x'] = self.aruco_markers['Depth_x']- self.calib.s_left
            self.aruco_markers['box_y'] = self.calib.s_height - self.aruco_markers['Depth_y'] - self.calib.s_bottom
            for j in self.aruco_markers.index:
                self.aruco_markers['is_inside_box'].loc[j] = self.calib.s_frame_width > (self.aruco_markers['Depth_x'].loc[j] - self.calib.s_left) and \
                                                  (self.aruco_markers['Depth_x'].loc[j] - self.calib.s_left)> 0 and \
                                                  (self.calib.s_frame_height > (self.calib.s_height - self.aruco_markers['Depth_y'].loc[j] - self.calib.s_bottom) and \
                                                  (self.calib.s_height - self.aruco_markers['Depth_y'].loc[j] - self.calib.s_bottom) > 0)

    def aruco_detect(self, image):
        """ Function to detect one aruco marker in a color image
        :param:
            image: numpy array containing a color image (BGR type)
        :return:
            corners: x, y location of a detected aruco marker(detect the 4 croners of the aruco)
            ids: id of the detected aruco
            rejectedImgPoints: show x, y coordinates of searches for aruco markers but not succesfull
       """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(self.aruco_dict)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        return corners, ids, rejectedImgPoints

    def get_location_marker(self, corners):
        """Get the middle position from the detected corners
         :param:
             corners: List containing the position x, y of the aruco marker
         :return:
             pr1: x location
             pr2: y location
        """

        pr1 = int(numpy.mean(corners[:, 0]))
        pr2 = int(numpy.mean(corners[:, 1]))
        return pr1, pr2

    def find_markers_ir(self, amount=None):
        """ Function to search for a determined amount of arucos in the infrared image. It will continue searching in
        different frames of the image until it finds all the markers
        :param:
            amount: specify the number of arucos to search
        :return:
            ir_marker: DataFrame with the id, x, y coordinates for the location of the aruco
                        And rotation and translation vectors for the pos estimation
        """
        labels = {'ids', 'Corners_IR_x', 'Corners_IR_y', "Rotation_vector", "Translation_vector"}
        df = pd.DataFrame(columns=labels)

        if amount is not None:
            while len(df) < amount:

                minim = 0
                maxim = numpy.arange(1000, 30000, 500)
                IR = self.kinect.get_ir_frame_raw()
                for i in maxim:
                    ir_use = numpy.interp(IR, (minim, i), (0, 255)).astype('uint8')
                    ir3 = numpy.stack((ir_use, ir_use, ir_use), axis=2)
                    corners, ids, rejectedImgPoints = self.aruco_detect(ir3)

                    if not ids is None:
                        for j in range(len(ids)):
                            if ids[j] not in df.ids.values:
                                rvec, tvec, trash = aruco.estimatePoseSingleMarkers([corners[j][0]],
                                                                                    self.size_of_marker,
                                                                                    self.mtx, self.dist)
                                x_loc, y_loc = self.get_location_marker(corners[j][0])
                                df_temp = pd.DataFrame(
                                    {'ids': [ids[j][0]], 'Corners_IR_x': [x_loc], 'Corners_IR_y': [y_loc],
                                     "Rotation_vector": [rvec], "Translation_vector": [tvec]})
                                df = pd.concat([df, df_temp], sort=False)

        self.ir_markers = df.reset_index(drop=True)
        return self.ir_markers

    def find_markers_rgb(self, amount=None):
        """ Function to search for a determined amount of arucos in the color image. It will continue searching in
        different frames of the image until it finds all the markers
        :param:
            amount: specify the number of arucos to search
        :return:
            rgb_markers: DataFrame with the id, x, y coordinates for the location of the aruco
                        and rotation and translation vectors for the pos estimation
        """

        labels = {"ids", "Corners_RGB_x", "Corners_RGB_y", "Rotation_vector", "Translation_vector"}
        df = pd.DataFrame(columns=labels)

        if amount is not None:
            while len(df) < amount:
                color = self.kinect.get_color()
                #color = color[self.kinect.calib.s_bottom:-self.kinect.calib.s_top, self.kinect.calib.s_left:-self.kinect.calib.s_right]
                corners, ids, rejectedImgPoints = self.aruco_detect(color)

                if not ids is None:
                    for j in range(len(ids)):
                        if ids[j] not in df.ids.values:
                            rvec, tvec, trash = aruco.estimatePoseSingleMarkers([corners[j][0]], self.size_of_marker,
                                                                                  self.mtx, self.dist)
                            x_loc, y_loc = self.get_location_marker(corners[j][0])
                            df_temp = pd.DataFrame(
                                {"ids": [ids[j][0]], "Corners_RGB_x": [x_loc], "Corners_RGB_y": [y_loc],
                                 "Rotation_vector": [rvec], "Translation_vector": [tvec]})
                            df = pd.concat([df, df_temp], sort=False)

        self.rgb_markers = df.reset_index(drop=True)
        return self.rgb_markers

    def find_markers_projector(self, amount=None):
        """ Function to search for a determined amount of arucos in the projected image. It will continue searching in
        different frames of the image until it finds all the markers
        :param:
            amount: specify the number of arucos to search
        :return:
            projector_markers: DataFrame with the id, x, y coordinates for the location of the aruco
                                and rotation and translation vectors for the pos estimation
            corner_middle: list that include the location of the central corner aruco with id=20
        """

        labels = {"ids", "Corners_projector_x", "Corners_projector_y", "Rotation_vector", "Translation_vector"}
        df = pd.DataFrame(columns=labels)

        if amount is not None:
            while len(df) < amount:
                color = self.kinect.get_color()
                corners, ids, rejectedImgPoints = self.aruco_detect(color)

                if ids is not None:
                    for j in range(len(ids)):
                        if ids[j] == 20:
                            # predefined id value to coincide with the projected aruco for the automatic calibration
                            # method used to calculate the scaling factor
                            self.corner_middle = corners[j][0]
                        if ids[j] not in df.ids.values:
                            rvec, tvec, trash = aruco.estimatePoseSingleMarkers([corners[j][0]], self.size_of_marker,
                                                                                self.mtx, self.dist)
                            x_loc, y_loc = self.get_location_marker(corners[j][0])
                            df_temp = pd.DataFrame(
                                {"ids": [ids[j][0]], "Corners_projector_x": [x_loc], "Corners_projector_y": [y_loc],
                                 "Rotation_vector": [rvec], "Translation_vector": [tvec]})
                            df = pd.concat([df, df_temp], sort=False)

        self.projector_markers = df.reset_index(drop=True)

        return self.projector_markers, self.corner_middle

    def create_CoordinateMap(self):
        """ Function to create a point to point map of the spatial/pixel equivalences between the depth space, color space and
        camera space. This method requires the depth frame to assign a depth value to the color point.
        :return:
            CoordinateMap: DataFrame with the x,y,z values of the depth frame; x,y equivalence between the depth space to camera space and
            real world values of x,y and z in meters
        """
        height, width = self.kinect.get_frame().shape
        x = numpy.arange(0, width)
        y = numpy.arange(0, height)
        xx, yy = numpy.meshgrid(x, y)
        xy_points = numpy.vstack([xx.ravel(), yy.ravel()]).T
        depth = self.kinect.get_frame()
        depth_x = []
        depth_y = []
        depth_z = []
        camera_x = []
        camera_y = []
        camera_z = []
        color_x = []
        color_y = []
        for i in range(len(xy_points)):
            x_point = xy_points[i, 0]
            y_point = xy_points[i, 1]
            z_point = depth[y_point][x_point]
            if z_point != 0:   # values that do not have depth information cannot be projected to the color space
                point = PyKinectV2._DepthSpacePoint(x_point, y_point)
                col = self.kinect.device._mapper.MapDepthPointToColorSpace(point, z_point)
                cam = self.kinect.device._mapper.MapDepthPointToCameraSpace(point, z_point)
                # since the position of the camera and sensor are different, they will not have the same coverage. Specially in the extremes
                if col.y > 0:
                    depth_x.append(x_point)
                    depth_y.append(y_point)
                    depth_z.append(z_point)
                    camera_x.append(cam.x)
                    camera_y.append(cam.y)
                    camera_z.append(cam.z)
                    color_x.append(int(col.x)+self.correction_x) ####TODO: constants addded since image is not exact when doing the transformation
                    color_y.append(int(col.y)-self.correction_y)

        self.CoordinateMap = pd.DataFrame({'Depth_x': depth_x,
                                           'Depth_y': depth_y,
                                           'Depth_Z(mm)': depth_z,
                                           'Color_x': color_x,
                                           'Color_y': color_y,
                                           'Camera_x(m)': camera_x,
                                           'Camera_y(m)': camera_y,
                                           'Camera_z(m)': camera_z})

        return self.CoordinateMap

    def create_aruco_marker(self, id = 1, resolution = 50, show=False, save=False):
        """ Function that creates a single aruco marker providing its id and resolution
        :param:
            id: int indicating the id of the aruco to create
            resolution: int
            show: boolean. Display the created aruco marker
            save: boolean. save the created aruco marker as an image "Aruco_Markers.jpg"
        :return:
            ArucoImage: numpy array with the aruco information
        """
        self.ArucoImage = 0

        aruco_dictionary = aruco.Dictionary_get(self.aruco_dict)
        img = aruco.drawMarker(aruco_dictionary, id, resolution)
        if show is True:
            plt.imshow(img, cmap=plt.cm.gray, interpolation="nearest")
            plt.axis("off")
        else:
            plt.close()

        if save is True:
            plt.savefig("Aruco_Markers.png")

        self.ArucoImage = img
        return self.ArucoImage

    def create_arucos_pdf(self, nx = 5, ny = 5, resolution = 50):
        aruco_dictionary = aruco.Dictionary_get(self.aruco_dict)
        fig = plt.figure()
        for i in range(1, nx * ny + 1):
            ax = fig.add_subplot(ny, nx, i)
            img = aruco.drawMarker(aruco_dictionary, i, resolution)
            plt.imshow(img, cmap='gray')
            plt.imshow(img, cmap='gray')
            ax.axis("off")
        plt.savefig("markers.pdf")
        plt.show()

    def plot_aruco_location(self, string_kind = 'RGB'):
        """ Function to visualize the location of the detected aruco markers in the image.
        :param:
            string_kind: IR -> Infrarred detection of aruco and visualization in infrared image
                         RGB -> Detection of aruco in color space and visualization as color image
                         Projector -> Detection of projected arucos inside sandbox and visualization in color image
        :return:
            image plot
        """
        plt.figure(figsize=(20, 20))
        if string_kind == 'IR':
            plt.imshow(self.kinect.get_ir_frame(), cmap="gray")
            plt.plot(self.ir_markers["Corners_IR_x"], self.ir_markers["Corners_IR_y"], "or")
            plt.show()
        elif string_kind == 'Projector':
            plt.imshow(self.kinect.get_color(), cmap="gray")
            plt.plot(self.projector_markers["Corners_projector_x"],
                     self.projector_markers["Corners_projector_y"], "or")
            plt.show()
        elif string_kind == 'RGB':
            #color = self.kinect.get_color()
            #color = color[self.kinect.calib.s_bottom:-self.kinect.calib.s_top,
             #       self.kinect.calib.s_left:-self.kinect.calib.s_right]
            plt.imshow(self.kinect.get_color())
            plt.plot(self.rgb_markers["Corners_RGB_x"], self.rgb_markers["Corners_RGB_y"], "or")
            plt.show()
        else:
            print('Select Type of projection -> IR, RGB or Projector')

    def convert_color_to_depth(self, ids, map, strg=None, data=None):
        """ Function to search in the previously created CoordinateMap - "create_CoordinateMap()" - the position of any
        detected aruco marker from the color space to the depth space.
        :param:
            strg: "Proj" or "Real". Select which type of aruco want to be converted
            ids: int. indicate the id of the aruco that want to be converted
            map: DataFrame. From the create_CoordinateMap() function
        :return:
            value: Return the line from the CoordinateMap DataFrame showing the equivalence of its position in the color
            space to the depth space
        """
        color_data = map[['Color_x', 'Color_y']]
        if strg is not None:
            if strg == 'Proj':
                rgb = self.projector_markers
                rgb2=rgb.loc[rgb['ids'] == ids]
                x_rgb = int(rgb2.Corners_projector_x.values)
                y_rgb = int(rgb2.Corners_projector_y.values)
            elif strg == 'Real':
                rgb = self.rgb_markers
                rgb2 = rgb.loc[rgb['ids'] == ids]
                x_rgb = int(rgb2.Corners_RGB_x.values)
                y_rgb = int(rgb2.Corners_RGB_y.values)

            distance = cdist([[x_rgb, y_rgb]], color_data)
            sorted_val = numpy.argsort(distance)[:][0]
            value = map.loc[sorted_val[0]]

        else:
            value = pd.DataFrame()
            if data is not None:
                for i in range(len(data)):
                    x_loc = data.loc[i].x
                    y_loc = data.loc[i].y

                    distance = cdist([[x_loc, y_loc]], color_data)
                    sorted_val = numpy.argsort(distance)[:][0]
                    value_i = pd.DataFrame(map.loc[sorted_val[0]]).T
                    value_i.insert(0, 'ids', data.loc[i].ids)
                    value = pd.concat([value, value_i], sort=False)

        return value

    def location_points(self, amount = None, plot = True):
        """ Function to search for a determined amount of arucos to introduce as a data point to the depth space
        :param:
            amount: specify the number of arucos to search
            plot: boolean to show the plot on color space and depth space if the mapped values are right
        :return:
            point_markers: DataFrame with the id, x, y coordinates for the location of the aruco
        """
        labels = {"ids", "x", "y"}
        df = pd.DataFrame(columns=labels)

        if amount is not None:
            while len(df) < amount:
                frame = self.kinect.get_color()
                color = frame#[self.rgb_markers.Corners_RGB_y.min():self.rgb_markers.Corners_RGB_y.max(),
                         #self.rgb_markers.Corners_RGB_x.min():self.rgb_markers.Corners_RGB_x.max()]
                corners, ids, rejectedImgPoints = self.aruco_detect(color)

                if ids is not None:
                    for j in range(len(ids)):
                        if ids[j] not in df.ids.values:
                            x_loc, y_loc = self.get_location_marker(corners[j][0])
                            df_temp = pd.DataFrame({"ids": [ids[j][0]], "x": [x_loc], "y": [y_loc]})
                            df = pd.concat([df, df_temp], sort=False)

        df = df.reset_index(drop=True)
        self.point_markers = self.convert_color_to_depth(None, self.CoordinateMap, data = df)

        self.point_markers =  self.point_markers.set_index(pd.Index(numpy.arange(len( self.point_markers))))

        if plot:
            color_crop = self.kinect.get_color()#[self.rgb_markers.Corners_RGB_y.min():self.rgb_markers.Corners_RGB_y.max(),
                         #self.rgb_markers.Corners_RGB_x.min():self.rgb_markers.Corners_RGB_x.max()]
            depth_crop = self.kinect.get_ir_frame()#[self.kinect.calib.s_bottom:-self.kinect.calib.s_top,
                         #self.kinect.calib.s_left:-self.kinect.calib.s_right]
            plt.figure(figsize=(20,20))
            plt.subplot(2, 1, 1)
            plt.imshow(color_crop)
            plt.plot(self.point_markers.Color_x, self.point_markers.Color_y, "or")
            #if self.rgb_markers.Corners_RGB_x.min() > 10:
            #    plt.xlim(self.rgb_markers.Corners_RGB_x.min(),self.rgb_markers.Corners_RGB_x.max())
            #    plt.ylim(self.rgb_markers.Corners_RGB_y.min(),self.rgb_markers.Corners_RGB_y.max())

            plt.subplot(2, 1, 2)
            plt.imshow(depth_crop)
            plt.plot(self.point_markers.Depth_x, self.point_markers.Depth_y, "or")
            plt.xlim(self.calib.s_left,depth_crop.shape[1]-self.calib.s_right)
            plt.ylim(depth_crop.shape[0]-self.calib.s_bottom, self.calib.s_top)
            plt.show()

        return self.point_markers

    def calibrate_camera_charucoBoard(self):
        '''
        Method to obtain the camera intrinsic parameters to perform the aruco pose estimation

        :return:
            mtx: cameraMatrix Output 3x3 floating-point camera matrix
            dist: Output vector of distortion coefficient
            rvecs: Output vector of rotation vectors (see Rodrigues ) estimated for each board view
            tvecs: Output vector of translation vectors estimated for each pattern view.
        '''

        aruco_dict = aruco.Dictionary_get(self.aruco_dict)
        board = aruco.CharucoBoard_create(7, 5, 1, .8, aruco_dict)
        images = []
        print('Start moving randomly the aruco board')
        n = 400 # number of frames
        for i in range(n):
            frame = self.kinect.get_color()
            images.append(frame)
        print("Stop moving the board")
        img_frame = numpy.array(images)[0::5]

        print("Calculating Aruco location of ",img_frame.shape[0], "images")
        allCorners = []
        allIds = []
        decimator = 0

        for im in img_frame:
            # print("=> Processing image {0}".format(im))
            # frame = cv2.imread(im)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            res = cv2.aruco.detectMarkers(gray, aruco_dict)

            if len(res[0]) > 0:
                res2 = cv2.aruco.interpolateCornersCharuco(res[0], res[1], gray, board)
                if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3 and decimator % 1 == 0:
                    allCorners.append(res2[1])
                    allIds.append(res2[2])

            decimator += 1
        imsize = gray.shape
        print("Finish")

        print("Calculating camera parameters")
        cameraMatrixInit = numpy.array([[2000., 0., imsize[0] / 2.],
                                     [0., 2000., imsize[1] / 2.],
                                     [0., 0., 1.]])

        distCoeffsInit = numpy.zeros((5, 1))
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL)
        ret, mtx, dist, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = cv2.aruco.calibrateCameraCharucoExtended(
            charucoCorners=allCorners,
            charucoIds=allIds,
            board=board,
            imageSize=imsize,
            cameraMatrix=cameraMatrixInit,
            distCoeffs=distCoeffsInit,
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

        print("Finish")

        self.calib.camera_mtx = mtx.tolist()
        self.calib.camera_dist = dist.tolist()

        return mtx, dist, rvecs, tvecs

    def real_time_poseEstimation(self):
        '''
        Method that display real time detection of the aruco markers with the pose estimation and id of each
        :return:
        '''
        cv2.namedWindow("Aruco")
        #frame = self.kinect.get_color()
        frame = self.kinect.get_color()#[270:900,640:1400]
        rval = True

        while rval:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            parameters = aruco.DetectorParameters_create()
            aruco_dict = aruco.Dictionary_get(self.aruco_dict)
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            if ids is not None:
                frame = aruco.drawDetectedMarkers(frame, corners, ids)
                # side lenght of the marker in meter
                rvecs, tvecs, trash = aruco.estimatePoseSingleMarkers(corners, self.size_of_marker, self.mtx, self.dist)
                for i in range(len(tvecs)):
                    frame = aruco.drawAxis(frame, self.mtx, self.dist, rvecs[i], tvecs[i], self.length_of_axis)

            cv2.imshow("Aruco", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            #frame = self.kinect.get_color()
            frame = self.kinect.get_color()#[270:900,640:1400]

            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                break

        cv2.destroyWindow("Aruco")

    def drawPoseEstimation(self, df, frame):
        '''
        Method that draws over the frame the coordinate system of each aruco marker in relation to the camera space
        :param
            df: data frame containing the information of the tranlation and rotation vectors previously detected
            frame: frame to draw the coordinate sytems
        :return:
            frame: with the resulting coordinate system

        '''
        for i in range(len(df)):
            frame = aruco.drawAxis(frame,
                               self.mtx,
                               self.dist,
                               df.loc[i].Rotation_vector[0],
                               df.loc[i].Translation_vector[0],
                               self.length_of_axis)
        return frame.get()

    def p_arucoMarker(self):
        """ Method to create an empty frame including 2 aruco markers.
        one in the upper left corner
        second one in the central part of the image.
        The id of the left-upper aruco is determined by the aruco position in the corner with resolution of 50
        The id in the center of the image is set to be 20 and resolution of 100
        :return.
            Frame as numpy array with the information of the aruco markers
        """
        width = self.calib.p_frame_width
        height = self.calib.p_frame_height

        # Creation of the aruco images as numpy array with size of resolution
        img_LU = self.create_aruco_marker(id=self.corner_id_LU, resolution= 50)
        img_c = self.create_aruco_marker(id=self.center_id, resolution= 100)

        # creation of empty numpy array with the size of the frame projected
        god = numpy.zeros((height, width))
        god.fill(255)

        # Placement of aruco markers in the image.
        # The Left uopper aruco will be placed with a constant offset distance in x and y from the corner
        god[height - img_LU.shape[0] - self.offset:height - self.offset, self.offset:img_LU.shape[1] + self.offset] =\
            numpy.flipud(img_LU)
        # The central aruco will be placed exactly in the middle of the image
        god[int(height / 2) - int(img_c.shape[0] / 2):int(height / 2) + int(img_c.shape[0] / 2),
        int(width / 2) - int(img_c.shape[0] / 2):int(width / 2) + int(img_c.shape[0] / 2)] = numpy.flipud(img_c)

        return god

    def move_image(self):
        """ Method to determine the distances between the aruco position in the corner of the sandbox in relation
        with the projected frame and the projected aruco marker.
        :return:
            p_frame_left: new value to update the calib.p_frame_left
            p_frame_top: new value to update the calib.p_frame_top
            p_frame_width: new value to update the calib.p_frame_width
            p_frame_height: new value to update the calib.p_frame_height
        """

        # Find the 2 corners of the projection
        df_p, corner = self.find_markers_projector(amount=2)
        # save the location of the aruco from the calibration file
        df_r = self.aruco_corners

        # extract the position x and y of the projected aruco
        x_p = int(df_p.loc[df_p.ids == self.corner_id_LU].Corners_projector_x.values)
        y_p = int(df_p.loc[df_p.ids == self.corner_id_LU].Corners_projector_y.values)

        # extract the position x and y of the corner sandbox where the projected aruco should be
        x_r = int(df_r.loc[df_r.ids == self.corner_id_LU].Color_x.values)
        y_r = int(df_r.loc[df_r.ids == self.corner_id_LU].Color_y.values)

        # scale factor using the resolution of the central aruco -> 100 pixels represented in reality
        cor = numpy.asarray(corner)
        scale_factor_x = 100 / (cor[:,0].max() - cor[:,0].min())
        scale_factor_y = 100 / (cor[:,1].max() - cor[:,1].min())

        # move x and y direction the whole frame to make coincide the projected aruco with the corner
        x_move = int(((x_p - x_r) * scale_factor_x)) - self.offset - self.pixel_displacement
        y_move = int(((y_p - y_r) * scale_factor_y)) - self.offset - self.pixel_displacement

        # provide with the location of the
        p_frame_left = self.calib.p_frame_left - x_move
        p_frame_top = self.calib.p_frame_top - y_move

        # Now same procedure with the center aruco by changing the width and height of the frame to make
        # coincide the center projected aruco with the center of the sandbox.
        x_c = df_r.Color_x.mean()
        y_c = df_r.Color_y.mean()

        x_pc = int(df_p.loc[df_p.ids == self.center_id].Corners_projector_x.values)
        y_pc = int(df_p.loc[df_p.ids == self.center_id].Corners_projector_y.values)

        width_move = int((x_c - x_pc) * scale_factor_x) + x_move - self.pixel_displacement
        height_move = int((y_c - y_pc) * scale_factor_y) + y_move - self.pixel_displacement

        p_frame_width = self.calib.p_frame_width + width_move
        p_frame_height = self.calib.p_frame_height + height_move

        return p_frame_left, p_frame_top, p_frame_width, p_frame_height

    def crop_image_aruco(self):
        """ Method that takes the location of the 4 real corners and crop the sensor extensions to this frame
        :return:
            s_top: new value to update the calib.s_top
            s_left: new value to update the calib.s_left
            s_bottom: new value to update the calib.s_bottom
            s_right: new value to update the calib.s_right
        """
        id_LU = self.aruco_corners.loc[self.aruco_corners.ids == self.corner_id_LU]
        id_DR = self.aruco_corners.loc[self.aruco_corners.ids == self.corner_id_DR]

        s_top = int(id_LU.Depth_y)
        s_left = int(id_LU.Depth_x)
        s_bottom = int(self.calib.s_height - id_DR.Depth_y)
        s_right = int(self.calib.s_width - id_DR.Depth_x)

        return s_top, s_left, s_bottom, s_right

