import os
from warnings import warn

# class infrastructre
from abc import ABCMeta, abstractmethod

try:
    import freenect

    warn('Two kernels cannot access the kinect at the same time. This will lead to a sudden death of the kernel. ' \
         'Be sure no other kernel is running before initialize a kinect object.', RuntimeWarning)
except ImportError:
    warn(
        'Freenect is not installed. if you are using the Kinect Version 2 on a windows machine, use the KinectV2 class!')

try:
    from pykinect2 import PyKinectV2  # try to import Wrapper for KinectV2 Windows SDK
    from pykinect2 import PyKinectRuntime

except ImportError:
    pass

try:
    import cv2
    from cv2 import aruco

except ImportError:
    # warn('opencv is not installed. Object detection will not work')
    pass

import webbrowser
import pickle
import numpy
import scipy
import scipy.ndimage

# for new projector
import panel as pn
from time import sleep

import logging

# for DummySensor
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata

#for resizing of block data:
import skimage

from itertools import count
from PIL import Image, ImageDraw
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib
# import gempy.hackathon as hackathon
import IPython
import threading

import json
import pandas as pd

from copy import copy

# TODO: When we move GeoMapModule import gempy just there
try:
    import gempy as gp

except ImportError:
    warn('gempy not found, GeoMap Module will not work')

### General settings and options ###

# logging and exception handling
verbose = False
if verbose:
    logging.basicConfig( filename="main.log",
                         filemode='w',
                         level=logging.WARNING,
                         format= '%(asctime)s - %(levelname)s - %(message)s',
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
        self.calibration = Calibration(self.calib, self.sensor, self.projector)


class Sensor:
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
        self.filter = filter # TODO: deprecate get_filtered_frame, make it switchable in runtime
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
        depth_array_masked = numpy.ma.masked_where(depth_array == 0, depth_array) # needed for V2?
        self.depth = numpy.ma.mean(depth_array_masked, axis=2)
        # apply gaussian filter
        self.depth = scipy.ndimage.filters.gaussian_filter(self.depth, self.sigma_gauss)

        return self.depth


class KinectV1(Sensor):

    # hard coded class attributes for KinectV1's native resolution
    name = 'kinect_v1'
    depth_width = 320
    depth_height = 240
    color_width = 640
    color_height = 480
    # TODO: Check!

    def setup(self):
        print("looking for kinect...")
        ctx = freenect.init()
        self.device = freenect.open_device(ctx, self.id)
        print(self.id)
        freenect.close_device(self.device)  # TODO Test if this has to be done!
        # get the first Depth frame already (the first one takes much longer than the following)
        self.depth = self.get_frame()
        print("kinect initialized")

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
        self.device = PyKinectRuntime.PyKinectRuntime(
            PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Infrared)
        self.depth = self.get_frame()
        self.color = self.get_color()
        #self.ir_frame_raw = self.get_ir_frame_raw()
        #self.ir_frame = self.get_ir_frame()

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
            ir_flattened.reshape((self.depth_height, self.depth_width)))  # reshape the array to 2D with native resolution of the kinectV2
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
        return self.color


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
        self.distance = numpy.sqrt(self.depth_width**2 + self.depth_height**2) * points_distance
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
        grid: Set of possible points to pick from
        n: desired number of points (without corners counting), not guaranteed to be reached
        distance: distance or range between points
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
            i = 4 # counter
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
            if cl < 1: break
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


class Calibration:
    """
    """

    def __init__(self, calibrationdata, sensor, projector, module):
        self.calib = calibrationdata
        self.sensor = sensor
        self.projector = projector
        self.module = module

        self.json_filename = None

        # customization
        self.c_under = '#DBD053'
        self.c_over = '#DB3A34'
        self.c_margin = '#084C61'
        self.c_margin_alpha = 0.5

        # init panel visualization
        pn.extension()

        ### projector widgets and links

        self._widget_p_frame_top = pn.widgets.IntSlider(name='Projector top margin',
                                                        value=self.calib.p_frame_top,
                                                        start=0,
                                                        end=self.calib.p_height - 20)
        self._widget_p_frame_top.link(self.projector.frame, callbacks={'value': self._callback_p_frame_top})


        self._widget_p_frame_left = pn.widgets.IntSlider(name='Projector left margin',
                                                         value=self.calib.p_frame_left,
                                                         start=0,
                                                         end=self.calib.p_width - 20)
        self._widget_p_frame_left.link(self.projector.frame, callbacks={'value': self._callback_p_frame_left})


        self._widget_p_frame_width = pn.widgets.IntSlider(name='Projector frame width',
                                                          value=self.calib.p_frame_width,
                                                          start=10,
                                                          end=self.calib.p_width)
        self._widget_p_frame_width.link(self.projector.frame, callbacks={'value': self._callback_p_frame_width})


        self._widget_p_frame_height = pn.widgets.IntSlider(name='Projector frame height',
                                                           value=self.calib.p_frame_height,
                                                           start=10,
                                                           end=self.calib.p_height)
        self._widget_p_frame_height.link(self.projector.frame, callbacks={'value': self._callback_p_frame_height})

        ### sensor widgets and links

        self._widget_s_top = pn.widgets.IntSlider(name='Sensor top margin',
                                                  bar_color=self.c_margin,
                                                  value=self.calib.s_top,
                                                  start=0,
                                                  end=self.calib.s_height)
        self._widget_s_top.link(self.plot_sensor, callbacks={'value': self._callback_s_top})

        self._widget_s_right = pn.widgets.IntSlider(name='Sensor right margin',
                                                    bar_color=self.c_margin,
                                                    value=self.calib.s_right,
                                                    start=0,
                                                    end=self.calib.s_width)
        self._widget_s_right.link(self.plot_sensor, callbacks={'value': self._callback_s_right})

        self._widget_s_bottom = pn.widgets.IntSlider(name='Sensor bottom margin',
                                                     bar_color=self.c_margin,
                                                     value=self.calib.s_bottom,
                                                     start=0,
                                                     end=self.calib.s_height)
        self._widget_s_bottom.link(self.plot_sensor, callbacks={'value': self._callback_s_bottom})

        self._widget_s_left = pn.widgets.IntSlider(name='Sensor left margin',
                                                   bar_color=self.c_margin,
                                                   value=self.calib.s_left,
                                                   start=0,
                                                   end=self.calib.s_width)
        self._widget_s_left.link(self.plot_sensor, callbacks={'value': self._callback_s_left})

        self._widget_s_min = pn.widgets.IntSlider(name='Vertical minimum',
                                                  bar_color=self.c_under,
                                                  value=self.calib.s_min,
                                                  start=0,
                                                  end=2000)
        self._widget_s_min.link(self.plot_sensor, callbacks={'value': self._callback_s_min})

        self._widget_s_max = pn.widgets.IntSlider(name='Vertical maximum',
                                                  bar_color=self.c_over,
                                                  value=self.calib.s_max,
                                                  start=0,
                                                  end=2000)
        self._widget_s_max.link(self.plot_sensor, callbacks={'value': self._callback_s_max})

        # refresh button

        self._widget_refresh_frame = pn.widgets.Button(name='Refresh sensor frame\n(3 sec. delay)!')
        self._widget_refresh_frame.param.watch(self._callback_refresh_frame, 'clicks', onlychanged=False)

        # save selection

        # Only for reading files --> Is there no location picker in panel widgets???
        #self._widget_json_location = pn.widgets.FileInput(name='JSON location')
        self._widget_json_filename = pn.widgets.TextInput(name='Choose a calibration filename:')
        self._widget_json_filename.param.watch(self._callback_json_filename, 'value', onlychanged=False)
        self._widget_json_filename.value = 'calibration.json'

        self._widget_json_save = pn.widgets.Button(name='Save calibration')
        self._widget_json_save.param.watch(self._callback_json_save, 'clicks', onlychanged=False)

        # sensor calibration visualization

        self.frame = self.sensor.get_filtered_frame()

        self.fig = plt.figure()
        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        self.fig.add_axes(self.ax)
        # close figure to prevent inline display
        plt.close(self.fig)
        self.plot_sensor()

        self.pn_fig = pn.pane.Matplotlib(self.fig, tight=False)

    ### layouts

    def calibrate_projector(self):
        widgets = pn.WidgetBox(self._widget_p_frame_top,
                               self._widget_p_frame_left,
                               self._widget_p_frame_width,
                               self._widget_p_frame_height)
        panel = pn.Column("### Map positioning", widgets)
        return panel

    def calibrate_sensor(self):
        widgets = pn.WidgetBox(self._widget_s_top,
                               self._widget_s_right,
                               self._widget_s_bottom,
                               self._widget_s_left,
                               self._widget_s_min,
                               self._widget_s_max,
                               self._widget_refresh_frame
                              )
        rows = pn.Row(widgets, self.pn_fig)
        panel = pn.Column("### Sensor calibration", rows)
        return panel

    def calibrate(self):
        tabs = pn.Tabs(('Projector', self.calibrate_projector()),
                       ('Sensor', self.calibrate_sensor()),
                       ('Save', pn.WidgetBox(self._widget_json_filename,
                                             self._widget_json_save))
                      )
        return tabs

    ### projector callbacks

    def _callback_p_frame_top(self, target, event):
        self.module.pause()
        # set value in calib
        self.calib.p_frame_top = event.new
        m = target.margin
        n = event.new
        # just changing single indices does not trigger updating of pane
        target.margin = [n, m[1], m[2], m[3]]
        self.module.resume()

    def _callback_p_frame_left(self, target, event):
        self.module.pause()
        self.calib.p_frame_left = event.new
        m = target.margin
        n = event.new
        target.margin = [m[0], m[1], m[2], n]
        self.module.resume()

    def _callback_p_frame_width(self, target, event):
        self.module.pause()
        self.calib.p_frame_width = event.new
        target.width = event.new
        target.param.trigger('object')
        self.module.resume()

    def _callback_p_frame_height(self, target, event):
        self.module.pause()
        self.calib.p_frame_height = event.new
        target.height = event.new
        target.param.trigger('object')
        self.module.resume()

    ### sensor callbacks

    def _callback_s_top(self, target, event):
        self.module.pause()
        # set value in calib
        self.calib.s_top = event.new
        # change plot and trigger panel update
        self.plot_sensor()
        self.pn_fig.param.trigger('object')
        self.module.resume()

    def _callback_s_right(self, target, event):
        self.module.pause()
        self.calib.s_right = event.new
        self.plot_sensor()
        self.pn_fig.param.trigger('object')
        self.module.resume()

    def _callback_s_bottom(self, target, event):
        self.module.pause()
        self.calib.s_bottom = event.new
        self.plot_sensor()
        self.pn_fig.param.trigger('object')
        self.module.resume()

    def _callback_s_left(self, target, event):
        self.module.pause()
        self.calib.s_left = event.new
        self.plot_sensor()
        self.pn_fig.param.trigger('object')
        self.module.resume()

    def _callback_s_min(self, target, event):
        self.module.pause()
        self.calib.s_min = event.new
        self.plot_sensor()
        self.pn_fig.param.trigger('object')
        self.module.resume()

    def _callback_s_max(self, target, event):
        self.module.pause()
        self.calib.s_max = event.new
        self.plot_sensor()
        self.pn_fig.param.trigger('object')
        self.module.resume()

    def _callback_refresh_frame(self, event):
        self.module.pause()
        sleep(3)
        self.frame = self.sensor.get_filtered_frame()
        self.plot_sensor()
        self.pn_fig.param.trigger('object')
        self.module.resume()

    def _callback_json_filename(self, event):
        self.json_filename = event.new

    def _callback_json_save(self, event):
        if self.json_filename is not None:
            self.calib.save_json(file=self.json_filename)

    ### other methods

    def plot_sensor(self):
        # clear old axes
        self.ax.cla()

        cmap = copy(plt.cm.gray)
        cmap.set_under(self.c_under, 1.0)
        cmap.set_over(self.c_over, 1.0)

        rec_t = plt.Rectangle((0, self.calib.s_height - self.calib.s_top), self.calib.s_width,
                              self.calib.s_top, fc=self.c_margin, alpha=self.c_margin_alpha)
        rec_r = plt.Rectangle((self.calib.s_width - self.calib.s_right, 0), self.calib.s_right,
                              self.calib.s_height, fc=self.c_margin, alpha=self.c_margin_alpha)
        rec_b = plt.Rectangle((0, 0), self.calib.s_width, self.calib.s_bottom, fc=self.c_margin, alpha=self.c_margin_alpha)
        rec_l = plt.Rectangle((0, 0), self.calib.s_left, self.calib.s_height, fc=self.c_margin, alpha=self.c_margin_alpha)

        self.ax.pcolormesh(self.frame, vmin=self.calib.s_min, vmax=self.calib.s_max, cmap=cmap)
        self.ax.add_patch(rec_t)
        self.ax.add_patch(rec_r)
        self.ax.add_patch(rec_b)
        self.ax.add_patch(rec_l)

        self.ax.set_axis_off()

        return True


class Projector:

    dpi = 100 # make sure that figures can be displayed pixel-precise

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

    def __init__(self, calibrationdata):
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

        if self.enable_legend:
            self.legend = pn.Column("### Legend",
                                    # add parameters from calibration for positioning
                                    width = 100,
                                    height = 100,
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

    def start_server(self):
        # TODO: Add specific port? port=4242
        # Check for instances and close them?
        self.panel.show(threaded=False)

    def show(self, figure):
        self.frame.object = figure
        #plt.close()

    def trigger(self):
        self.frame.param.trigger('object')


class CalibrationData:
    """

    """

    def __init__(self,
                 p_width=800, p_height=600, p_frame_top=0, p_frame_left=0,
                 p_frame_width=600, p_frame_height=450,
                 s_top=0, s_right=0, s_bottom=0, s_left=0, s_min=700, s_max=1500,
                 file=None):
        """

        Args:
            p_width=800
            p_height=600
            p_frame_top=0
            p_frame_left=0
            p_frame_width=600
            p_frame_height=450
            s_top=0
            s_right=0
            s_bottom=0
            s_left=0
            s_min=700
            s_max=1500
            file=None

        Returns:
            None

        """

        # version identifier (will be changed if new calibration parameters are introduced / removed)
        self.version = "0.8alpha"

        # projector
        self.p_width = p_width
        self.p_height = p_height

        self.p_frame_top = p_frame_top
        self.p_frame_left = p_frame_left
        self.p_frame_width = p_frame_width
        self.p_frame_height = p_frame_height

        #self.p_legend_top =
        #self.p_legend_left =
        #self.p_legend_width =
        #self.p_legend_height =

        # hot area
        #self.p_hot_top =
        #self.p_hot_left =
        #self.p_hot_width =
        #self.p_hot_height =

        # profile area
        #self.p_profile_top =
        #self.p_profile_left =
        #self.p_profile_width =
        #self.p_profile_height =

        # sensor (e.g. Kinect)
        self.s_name = 'generic' # name to identify the associated sensor device
        self.s_width = 500 # will be updated by sensor init
        self.s_height = 400 # will be updated by sensor init

        self.s_top = s_top
        self.s_right = s_right
        self.s_bottom = s_bottom
        self.s_left = s_left
        self.s_min = s_min
        self.s_max = s_max

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

class Scale:
    """
    class that handles the scaling of whatever the sandbox shows and the real world sandbox
    self.extent: 3d extent of the model in the sandbox in model units.

    """

    def __init__(self, calibration: Calibration = None, xy_isometric=True, extent=None):
        """

        Args:
            calibration:
            xy_isometric:
            extent:
        """
        self.calibration = calibration
        """
        if isinstance(calibration, Calibration):
            self.calibration = calibration
        else:
            raise TypeError("you must pass a valid calibration instance")
        """
        self.xy_isometric = xy_isometric
        self.scale = [None, None, None]
        self.pixel_size = [None, None]
        self.pixel_scale = [None, None]
        self.output_res = None

        if extent is None:  # extent should be array with shape (6,) or convert to list?
            self.extent = numpy.asarray([
                0.0,
                self.calibration.calibration_data.box_width,
                0.0,
                self.calibration.calibration_data.box_height,
                self.calibration.calibration_data.z_range[0],
                self.calibration.calibration_data.z_range[1],
            ])

        else:
            self.extent = numpy.asarray(extent)  # check: array with 6 entries!

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

        self.output_res = (self.calibration.calibration_data.x_lim[1] -
                           self.calibration.calibration_data.x_lim[0],
                           self.calibration.calibration_data.y_lim[1] -
                           self.calibration.calibration_data.y_lim[0])
        self.pixel_scale[0] = float(self.extent[1] - self.extent[0]) / float(self.output_res[0])
        self.pixel_scale[1] = float(self.extent[3] - self.extent[2]) / float(self.output_res[1])
        self.pixel_size[0] = float(self.calibration.calibration_data.box_width) / float(self.output_res[0])
        self.pixel_size[1] = float(self.calibration.calibration_data.box_height) / float(self.output_res[1])

        # TODO: change the extrent in place!! or create a new extent object that stores the extent after that modification.
        if self.xy_isometric == True:  # model is extended in one horizontal direction to fit  into box while the scale
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
        self.scale[2] = float(self.extent[5] - self.extent[4]) / (
                self.calibration.calibration_data.z_range[1] -
                self.calibration.calibration_data.z_range[0])
        print("scale in Model units/ mm (X,Y,Z): " + str(self.scale))

    # TODO: manually define zscale and either lower or upper limit of Z, adjust rest accordingly.


class Grid:
    """
    class for grid objects. a grid stores the 3D coordinate of each pixel recorded by the kinect in model coordinates
    a calibration object must be provided, it is used to crop the kinect data to the area of interest
    TODO:  The cropping should be done in the kinect class, with calibration_data passed explicitly to the method! Do this for all the cases where calibration data is needed!
    """

    def __init__(self, calibration=None, scale=None, ):
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
            self.scale = Scale(calibration=self.calibration)
            print("no scale provided or scale invalid. A default scale instance is used")
        self.empty_depth_grid = None
        self.create_empty_depth_grid()

    def create_empty_depth_grid(self):
        """
        Sets up XY grid (Z is empty, that is where the name is coming from)

        Returns:

        """

        grid_list = []
        self.output_res = (self.calibration.calibration_data.x_lim[1] -
                           self.calibration.calibration_data.x_lim[0],
                           self.calibration.calibration_data.y_lim[1] -
                           self.calibration.calibration_data.y_lim[0])
        """compare:
        for x in range(self.output_res[1]):
            for y in range(self.output_res[0]):
                grid_list.append([y * self.scale.pixel_scale[1] + self.scale.extent[2], x * self.scale.pixel_scale[0] + self.scale.extent[0]])
        """

        for y in range(self.output_res[1]):
            for x in range(self.output_res[0]):
                grid_list.append([x * self.scale.pixel_scale[0] + self.scale.extent[0],
                                  y * self.scale.pixel_scale[1] + self.scale.extent[2]])

        empty_depth_grid = numpy.array(grid_list)
        self.empty_depth_grid = empty_depth_grid
        self.depth_grid = None  # I know, this should have thew right type.. anyway.
        print("the shown extent is [" + str(self.empty_depth_grid[0, 0]) + ", " +
              str(self.empty_depth_grid[-1, 0]) + ", " +
              str(self.empty_depth_grid[0, 1]) + ", " +
              str(self.empty_depth_grid[-1, 1]) + "] "
              )

        # return self.empty_depth_grid

    def update_grid(self, depth):
        """
        Appends the z (depth) coordinate to the empty depth grid.
        this has to be done every frame while the xy coordinates only change if the calibration or model extent is changed.
        For performance reasons these steps are therefore separated.

        Args:
            depth:

        Returns:

        """

        # TODO: is this flip still necessary?
        depth = numpy.fliplr(depth)  ##dirty workaround to get the it running with new gempy version.
        filtered_depth = numpy.ma.masked_outside(depth, self.calibration.calibration_data.z_range[0],
                                                 self.calibration.calibration_data.z_range[1])
        scaled_depth = self.scale.extent[5] - (
                (filtered_depth - self.calibration.calibration_data.z_range[0]) / (
                self.calibration.calibration_data.z_range[1] -
                self.calibration.calibration_data.z_range[0]) * (self.scale.extent[5] - self.scale.extent[4]))
        rotated_depth = scipy.ndimage.rotate(scaled_depth, self.calibration.calibration_data.rot_angle,
                                             reshape=False)
        cropped_depth = rotated_depth[self.calibration.calibration_data.y_lim[0]:
                                      self.calibration.calibration_data.y_lim[1],
                        self.calibration.calibration_data.x_lim[0]:
                        self.calibration.calibration_data.x_lim[1]]

        flattened_depth = numpy.reshape(cropped_depth, (numpy.shape(self.empty_depth_grid)[0], 1))
        depth_grid = numpy.concatenate((self.empty_depth_grid, flattened_depth), axis=1)

        self.depth_grid = depth_grid


class Contour:  # TODO: change the whole thing to use keyword arguments!! Move to Plot class!
    """
    class to handle contour lines in the sandbox. contours can show depth or anything else.
    TODO: pass on keyword arguments to the plot and label functions for more flexibility

    """

    def __init__(self, start, end, step, show=True, show_labels=False, linewidth=1.0, colors=[(0, 0, 0, 1.0)],
                 inline=0, fontsize=15, label_format='%3.0f'):
        """

        Args:
            start:
            end:
            step:
            show:
            show_labels:
            linewidth:
            colors:
            inline:
            fontsize:
            label_format:

        Returns:
            None

        """
        self.start = start
        self.end = end
        self.step = step
        self.show = show
        self.show_labels = show_labels
        self.linewidth = linewidth
        self.colors = colors
        self.levels = numpy.arange(self.start, self.end, self.step)
        self.contours = None
        self.data = None  # Data has to be updated for each frame

        # label attributes:
        self.inline = inline
        self.fontsize = fontsize
        self.label_format = label_format


class Plot:
    """
    handles the plotting of a sandbox model

    """

    dpi = 100 # make sure that figures can be displayed pixel-precise

    def __init__(self, calibrationdata, contours=True, patches=False,
                 cmap=None, over=None, under=None, bad=None,
                 norm=None, lot=None, c_patch='#084C61', c_patch_alpha=0.5,
                 contours_step=10, contours_width=1.0, contours_color=[(0, 0, 0, 1.0)],
                 contours_label=False, contours_label_inline=0,
                 contours_label_fontsize=15, contours_label_format='%3.0f'):

        self.calib = calibrationdata

        ### flags
        self.contours = contours
        self.patches = patches

        ### pcolormesh setup
        self.cmap = plt.cm.get_cmap(cmap)
        if over is not None: self.cmap.set_over(over, 1.0)
        if under is not None: self.cmap.set_under(under, 1.0)
        if bad is not None: self.cmap.set_bad(bad, 1.0)

        self.norm = norm
        self.lot = lot

        ### contours setup
        self.contours_levels = numpy.arange(self.calib.s_min, self.calib.s_max, contours_step)
        self.contours_width = contours_width
        self.contours_color = contours_color
        self.contours_label = contours_label
        self.contours_label_inline = contours_label_inline
        self.contours_label_fontsize = contours_label_fontsize
        self.contours_label_format = contours_label_format

        ### patches_setup
        self.c_patch = c_patch
        self.c_patch_alpha = c_patch_alpha

        # save the figure's Matplotlib number to recall
        #self.number = None
        self.figure = None
        self.ax = None # current plot composition
        # initial figure for starting projector
        self.create_empty_frame()

    def create_empty_frame(self):
        self.figure = plt.figure(figsize=(self.calib.p_frame_width / self.dpi, self.calib.p_frame_height / self.dpi),
                                 dpi=self.dpi)
        self.ax = plt.Axes(self.figure, [0., 0., 1., 1.])
        self.figure.add_axes(self.ax)

        # close figure to prevent inline display
        plt.close(self.figure)
        self.ax.set_axis_off()

        return True

    def render_frame(self, data):
        self.ax.cla()  # clear axes to draw new ones on figure
        self.ax.pcolormesh(data, vmin=self.calib.s_min, vmax=self.calib.s_max, cmap=self.cmap, norm=self.norm)

        if self.contours:
            self.add_contours(data)

        if self.patches:
            self.add_patches()

        # crop axis to input data dimensions, e.g. when labels are out of axes
        self.ax.axis([0, data.shape[1], 0, data.shape[0]])
        self.ax.set_axis_off()

        return True

    def add_contours(self, data):
        """
        renders contours to the current plot object. \
        The data has to come in a specific shape as needed by the matplotlib contour function.
        we explicity enforce to provide X and Y at this stage (you are welcome to change this)

        Args:
            data:  a list with the form x,y,z
                x: list of the coordinates in x direction (e.g. range(Scale.output_res[0])
                y: list of the coordinates in y direction (e.g. range(Scale.output_res[1])
                z: 2D array-like with the values to be contoured

        self.contours_label = contours_label
        self.contours_width = contours_width
        self.contours_color = contours_color
        self.contours_levels = numpy.arange(self.calib.s_min, self.calib.s_max, contours_step)
        self.contours_label_inline = contours_inline
        self.contours_label_fontsize = contours_label_fontsize
        self.contours_label_format = contours_label_format
        Returns:

        """

        contours = self.ax.contour(data,
                                   levels=self.contours_levels,
                                   linewidths=self.contours_width,
                                   colors=self.contours_color)
        if self.contours_label is True:
            self.ax.clabel(contours,
                           inline=self.contours_label_inline,
                           fontsize=self.contours_label_fontsize,
                           fmt=self.contours_label_format)

    def add_patches(self):
        ''' Only usefull when uncropped frame is passed. '''

        w = self.calib.scale_factor[0]
        h = self.calib.scale_factor[1]

        rec_t = plt.Rectangle((0, self.calib.s_height - self.calib.s_top), self.calib.s_width, self.calib.s_top,
                              fc=self.c_patch, alpha=self.c_patch_alpha)
        rec_r = plt.Rectangle((self.calib.s_width - self.calib.s_right, 0), self.calib.s_right, self.calib.s_height,
                              fc=self.c_patch, alpha=self.c_patch_alpha)
        rec_b = plt.Rectangle((0, 0), self.calib.s_width, self.calib.s_bottom,
                              fc=self.c_patch, alpha=self.c_patch_alpha)
        rec_l = plt.Rectangle((0, 0), self.calib.s_left, self.calib.s_height,
                              fc=self.c_patch, alpha=self.c_patch_alpha)

        self.ax.add_patch(rec_t)
        self.ax.add_patch(rec_r)
        self.ax.add_patch(rec_b)
        self.ax.add_patch(rec_l)

    # def change_size(self):
    #     self. figure = plt.figure(num=self.figure.number, figsize=(self.calib.p_frame_width / self.dpi,
    #                                                                self.calib.p_frame_height / self.dpi))
    #
    #     # close figure to prevent inline display
    #     plt.close(self.figure)
    #
    #     return True
    #
    # def add_lith_contours(self, block, levels=None):
    #     """
    #
    #     Args:
    #         block:
    #         levels:
    #
    #     Returns:
    #
    #     """
    #     plt.contourf(block, levels=levels, cmap=self.cmap, norm=self.norm, extend="both")
    #
    # def create_legend(self):
    #     """ Returns:
    #     """
    #     pass

    def create_cmap(self, clist):
        cmap = matplotlib.colors.ListedColormap(clist)
        return cmap

    def create_norm(self, _min, _max):
        norm = matplotlib.colors.Normalize(vmin=_min, vmax=_max)
        return norm

class Module:
    """
    Parent Module with threading methods and abstract attributes and methods for child classes
    """
    __metaclass__ = ABCMeta

    def __init__(self, calibrationdata, sensor, projector, crop=True, **kwargs):
        self.calib = calibrationdata
        self.sensor = sensor
        self.projector = projector
        self.plot = Plot(self.calib, **kwargs)

        # flags
        self.crop = crop

        # threading
        self.lock = threading.Lock()
        self.thread = None
        self.thread_status = 'stopped' # status: 'stopped', 'running', 'paused'

        #self.setup()

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
            with self.lock:
                self.update()

    def run(self):
        if self.thread_status != 'running':
            self.thread_status = 'running'
            self.thread = threading.Thread(target=self.thread_loop, daemon=True, )
            self.thread.start()
            print('Thread started or resumed...')
        else:
            print('Thread already running.')

    def stop(self):
        self.thread_status = 'stopped'  # set flag to end thread loop
        self.thread.join()  # wait for the thread to finish
        print('Thread stopped.')

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

    def crop_frame(self, frame):
        crop = frame[self.calib.s_bottom:-self.calib.s_top, self.calib.s_left:-self.calib.s_right]
        clip = numpy.clip(crop, self.calib.s_min, self.calib.s_max)
        return clip


class TopoModule(Module):
    """
    Module for simple Topography visualization without computing a geological model
    """
    def setup(self):
        frame = self.sensor.get_filtered_frame()
        #if self.crop is True:
        #    frame = self.crop_frame(frame)
        self.plot.render_frame(frame)
        self.projector.frame.object = self.plot.figure

    def update(self):
        frame = self.sensor.get_filtered_frame()
        #if self.crop is True:
        #    frame = self.crop_frame(frame)
        self.plot.render_frame(frame)
        self.projector.trigger()


class CalibModule(Module):
    """
    Module for calibration and responsive visualization
    """

    def __init__(self,*args, **kwargs):

        # customization
        self.c_under = '#DBD053'
        self.c_over = '#DB3A34'
        self.c_margin = '#084C61'
        self.c_margin_alpha = 0.5

        # call parents' class init, use greyscale colormap as standard and extreme color labeling
        super().__init__(*args, contours=True, cmap='Greys_r', over=self.c_over, under=self.c_under, **kwargs)

        self.json_filename = None

        # sensor calibration visualization
        self.calib_frame = None
        self.calib_plot = Plot(self.calib, patches=True,
                               contours=True, cmap='Greys_r', over=self.c_over, under=self.c_under, **kwargs)

        # init panel visualization
        pn.extension()
        self._create_widgets()

    ### standard methods
    def setup(self):
        frame = self.sensor.get_filtered_frame()
        if self.crop:
            frame = self.crop_frame(frame)
        self.plot.render_frame(frame)
        self.projector.frame.object = self.plot.figure

        # sensor calibration visualization
        self.calib_frame = self.sensor.get_filtered_frame()
        self.calib_plot.render_frame(self.calib_frame)
        self.pn_figure.object = self.calib_plot.figure

    def update(self):
        frame = self.sensor.get_filtered_frame()
        if self.crop:
            frame = self.crop_frame(frame)
        self.plot.render_frame(frame)
        self.projector.trigger()

    def update_calib_plot(self):
        self.calib_plot.render_frame(self.calib_frame)
        self.pn_figure.param.trigger('object')

    ### layouts
    def calibrate_projector(self):
        widgets = pn.WidgetBox(self._widget_p_frame_top,
                               self._widget_p_frame_left,
                               self._widget_p_frame_width,
                               self._widget_p_frame_height)
        panel = pn.Column("### Map positioning", widgets)
        return panel

    def calibrate_sensor(self):
        widgets = pn.WidgetBox(self._widget_s_top,
                               self._widget_s_right,
                               self._widget_s_bottom,
                               self._widget_s_left,
                               self._widget_s_min,
                               self._widget_s_max,
                               self._widget_refresh_frame
                              )
        rows = pn.Row(widgets, self.pn_figure)
        panel = pn.Column("### Sensor calibration", rows)
        return panel

    def calibrate(self):
        tabs = pn.Tabs(('Projector', self.calibrate_projector()),
                       ('Sensor', self.calibrate_sensor()),
                       ('Save', pn.WidgetBox(self._widget_json_filename,
                                             self._widget_json_save))
                      )
        return tabs

    def _create_widgets(self):

        ### sensor calibration visualization

        self.pn_figure = pn.pane.Matplotlib(plt.figure(), tight=False)

        ### projector widgets and links

        self._widget_p_frame_top = pn.widgets.IntSlider(name='Projector top margin',
                                                        value=self.calib.p_frame_top,
                                                        start=0,
                                                        end=self.calib.p_height - 20)
        self._widget_p_frame_top.link(self.projector.frame, callbacks={'value': self._callback_p_frame_top})


        self._widget_p_frame_left = pn.widgets.IntSlider(name='Projector left margin',
                                                         value=self.calib.p_frame_left,
                                                         start=0,
                                                         end=self.calib.p_width - 20)
        self._widget_p_frame_left.link(self.projector.frame, callbacks={'value': self._callback_p_frame_left})


        self._widget_p_frame_width = pn.widgets.IntSlider(name='Projector frame width',
                                                          value=self.calib.p_frame_width,
                                                          start=10,
                                                          end=self.calib.p_width)
        self._widget_p_frame_width.link(self.projector.frame, callbacks={'value': self._callback_p_frame_width})


        self._widget_p_frame_height = pn.widgets.IntSlider(name='Projector frame height',
                                                           value=self.calib.p_frame_height,
                                                           start=10,
                                                           end=self.calib.p_height)
        self._widget_p_frame_height.link(self.projector.frame, callbacks={'value': self._callback_p_frame_height})

        ### sensor widgets and links

        self._widget_s_top = pn.widgets.IntSlider(name='Sensor top margin',
                                                  bar_color=self.c_margin,
                                                  value=self.calib.s_top,
                                                  start=0,
                                                  end=self.calib.s_height)
        self._widget_s_top.link(self.update_calib_plot, callbacks={'value': self._callback_s_top})

        self._widget_s_right = pn.widgets.IntSlider(name='Sensor right margin',
                                                    bar_color=self.c_margin,
                                                    value=self.calib.s_right,
                                                    start=0,
                                                    end=self.calib.s_width)
        self._widget_s_right.link(self.update_calib_plot, callbacks={'value': self._callback_s_right})

        self._widget_s_bottom = pn.widgets.IntSlider(name='Sensor bottom margin',
                                                     bar_color=self.c_margin,
                                                     value=self.calib.s_bottom,
                                                     start=0,
                                                     end=self.calib.s_height)
        self._widget_s_bottom.link(self.update_calib_plot, callbacks={'value': self._callback_s_bottom})

        self._widget_s_left = pn.widgets.IntSlider(name='Sensor left margin',
                                                   bar_color=self.c_margin,
                                                   value=self.calib.s_left,
                                                   start=0,
                                                   end=self.calib.s_width)
        self._widget_s_left.link(self.update_calib_plot, callbacks={'value': self._callback_s_left})

        self._widget_s_min = pn.widgets.IntSlider(name='Vertical minimum',
                                                  bar_color=self.c_under,
                                                  value=self.calib.s_min,
                                                  start=0,
                                                  end=2000)
        self._widget_s_min.link(self.update_calib_plot, callbacks={'value': self._callback_s_min})

        self._widget_s_max = pn.widgets.IntSlider(name='Vertical maximum',
                                                  bar_color=self.c_over,
                                                  value=self.calib.s_max,
                                                  start=0,
                                                  end=2000)
        self._widget_s_max.link(self.update_calib_plot, callbacks={'value': self._callback_s_max})

        # refresh button

        self._widget_refresh_frame = pn.widgets.Button(name='Refresh sensor frame\n(3 sec. delay)!')
        self._widget_refresh_frame.param.watch(self._callback_refresh_frame, 'clicks', onlychanged=False)

        # save selection

        # Only for reading files --> Is there no location picker in panel widgets???
        #self._widget_json_location = pn.widgets.FileInput(name='JSON location')
        self._widget_json_filename = pn.widgets.TextInput(name='Choose a calibration filename:')
        self._widget_json_filename.param.watch(self._callback_json_filename, 'value', onlychanged=False)
        self._widget_json_filename.value = 'calibration.json'

        self._widget_json_save = pn.widgets.Button(name='Save calibration')
        self._widget_json_save.param.watch(self._callback_json_save, 'clicks', onlychanged=False)

        return True

    ### projector callbacks

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

    ### sensor callbacks

    def _callback_s_top(self, target, event):
        self.pause()
        # set value in calib
        self.calib.s_top = event.new
        # change plot and trigger panel update
        self.update_calib_plot()
        self.resume()

    def _callback_s_right(self, target, event):
        self.pause()
        self.calib.s_right = event.new
        self.update_calib_plot()
        self.resume()

    def _callback_s_bottom(self, target, event):
        self.pause()
        self.calib.s_bottom = event.new
        self.update_calib_plot()
        self.resume()

    def _callback_s_left(self, target, event):
        self.pause()
        self.calib.s_left = event.new
        self.update_calib_plot()
        self.resume()

    def _callback_s_min(self, target, event):
        self.pause()
        self.calib.s_min = event.new
        self.update_calib_plot()
        self.resume()

    def _callback_s_max(self, target, event):
        self.pause()
        self.calib.s_max = event.new
        self.update_calib_plot()
        self.resume()

    def _callback_refresh_frame(self, event):
        self.pause()
        sleep(3)
        # only here get a new frame before updating the plot
        self.calib_frame = self.sensor.get_filtered_frame()
        self.update_calib_plot()
        self.resume()

    def _callback_json_filename(self, event):
        self.json_filename = event.new

    def _callback_json_save(self, event):
        if self.json_filename is not None:
            self.calib.save_json(file=self.json_filename)


class BlockModule(Module):
    # child class of Model

    def __init__(self, calibrationdata, sensor, projector, crop=True, **kwarg):
        Module.__init__(self, calibrationdata, sensor, projector, crop, **kwarg) #call parent init
        self.block_dict = {}
        self.cmap_dict = {}
        self.displayed_dataset_key = "Zone" # variable to choose displayed dataset in runtime
        self.rescaled_block_dict = {}
        self.index = None
        self.widget = None #widget to change models in runtime

    def setup(self):
        if self.block_dict is None:
            print("No model loaded. Load a model first with load_module_vip(infile)")
            pass
        elif self.cmap_dict is None:
            self.set_colormaps()
        self.rescale_blocks()

        self.projector.frame.object = self.plot.figure #Link figure to projector

    def update(self):
        with self._lock:
            frame = self.sensor.get_filtered_frame()
            if self.crop is True:
                frame = self.crop_frame(frame)

            data = self.rescaled_block_dict[self.displayed_dataset_key]
            zmin = self.calib.s_min
            zmax = self.calib.s_max

            index = (frame - zmin) / (zmax - zmin) * (data.shape[2] - 1.0)  # convert the z dimension to index
            index = index.round()  # round to next integer
            index = index.astype('int')

            # querry the array:
            i, j = numpy.indices(data[..., 0].shape)  # create arrays with the indices in x and y
            result = data[i, j, index]

            self.plot.ax.cla()

            # self.block = rasterdata.reshape((self.calib.s_frame_height, self.calib.s_frame_width))
            # self.ax.pcolormesh(self.block,
            self.plot.ax.pcolormesh(result, vmin=data.min(), vmax=data.max(), cmap=self.cmap_dict[self.displayed_dataset_key])
            #render and display
            self.plot.ax.axis([0, self.calib.s_frame_width, 0, self.calib.s_frame_height])
            self.plot.ax.set_axis_off()

            self.projector.trigger()
            return True

    def set_colormap(self, key=None, cmap='jet', norm=None):
        min = self.block_dict[key].min()
        max = self.block_dict[key].max()
        if cmap==None:
            cmap='jet'
        if norm==None:
            norm= Plot.create_norm(min, max)
        self.cmap_dict[key] = [cmap,norm]

    def set_colormaps(self, cmap='jet', norm=None):
        """
        iterates over all datasets and checks if a colormap has been set. if no colormaps exists it creates one.
        default colormap: jet
        :param cmap:
        :param norm:
        :return:
        """
        for key in self.block_dict.keys:
            min = self.block_dict[key].min()
            max = self.block_dict[key].max()
            if self.cmap_dict[key] is None:
                if norm is None:
                    norm=Plot.create_norm(min, max)
                self.cmap_dict[key]=[cmap, norm]



    def load_model_vip(self, infile):
        # parse the file
        f = open(infile, "r")

        while True:  # skip header
            l = f.readline().split()
            if len(l) > 2 and l[1] == "Size":
                print(l)
                break

        # n cells
        l = f.readline().split()
        nx = int(l[1])
        ny = int(l[2])
        nz = int(l[3])

        print(nx, ny, nz)

        while True:  # skip to Data section, ignoring Livecells
            l = f.readline()
            ls = l.split()
            if len(ls) == 3:
                if ls[2] == 'parameters':
                    break

        for i in range(4):
            f.readline()

        # parse the data

        while True: #iterate over all available blocks, stop at first error
            try:
                self.parse_block_VIP(f, self.block_dict, nx, ny, nz)
            except:
                break

        f.close() #close the file

    def rescale_blocks(self): #scale the blocks xy Size to the cropped size of the sensor
        for key in self.block_dict.keys():
            rescaled_block = skimage.transform.resize(
                self.block_dict[key],
                (self.calib.s_frame_height, self.calib.s_frame_width),
                order=0
            )
            self.rescaled_block_dict[key] = rescaled_block

    def clear_models(self):
        self.block_dict = {}

    def clear_rescaled_models(self):
        self.rescaled_block_dict = {}

    def clear_cmaps(self):
        self.cmap_dict = {}

    def load_single_block_file(self, filename, key, value_dict, nx, ny, nz, values_per_line=8 ):
        """
        Function to load a single block from a file that comes without any metadata.
        A string for the key as well as the dimension of the Block model have to be specified.

        If possible, all Data should be imported in a single VIP Text file. with load_model_vip()
        This method is then obsolete.

        :param filename: string to the file location on disc
        :param key: string under which the data will be accesible in the block_dict
        :param value_dict: hand over the block_dict the data will be stored in
        :param nx: number of cells in x
        :param ny: number of cells in y
        :param nz: number of cells in z
        :return: nothing, changes value_dict in place.

        """
        f = open(filename, "r")
        values = []
        data_np = numpy.empty((nx, ny, nz))


        # read block data
        blocklength = nx // values_per_line
        if (nx % values_per_line) != 0:
            blocklength = blocklength + 1

        for z in range(nz):
            for i in range(3):
                f.readline()
            for y in range(ny):
                x = 0

                for line in range(blocklength):
                    l = f.readline().split()

                    for i in range(values_per_line):
                        value = l[i]
                        # data.loc[x,y,z] = value
                        # values.append(value)
                        data_np[x, y, z] = float(value)
                        x = x + 1
                f.readline()

        value_dict[key] = data_np

    def parse_block_VIP(self, current_file, value_dict, nx, ny, nz):

        f = current_file

        # prepare storage objects
        key = f.readline().split()[0]
        values = []
        data_np = numpy.empty((nx, ny, nz))

        # skip to Beginning of block
        #   for i in range(4):
        #       print(f.readline())

        # read block data
        values_per_line = 8
        blocklength = nx // values_per_line
        if (nx % values_per_line) != 0:
            blocklength = blocklength + 1

        for z in range(nz):
            for i in range(3):
                f.readline()
            for y in range(ny):
                x = 0

                for line in range(blocklength):
                    l = f.readline().split()

                    for i in range(values_per_line):
                        value = l[i]
                        # data.loc[x,y,z] = value
                        # values.append(value)
                        data_np[x, y, z] = float(value)
                        x = x + 1
                f.readline()
        print('done')
        value_dict[key] = data_np

        # skip to end of block
        for i in range(4):
            f.readline()

    def show_selector(self):
        """
        displays a widget to toggle between the currently active dataset while the sandbox is running

        :return:
        """
        pn.extension()
        self.widget = pn.widgets.RadioButtonGroup(name='Model selector',
                                                  options=list(self.block_dict.keys()),
                                                  button_type='success')
        self.widget.param.watch(self.update_selection, 'value', onlychanged=False)
        return self.widget

    def update_selection(self, event):
        """
        callback function for the widget to update the self.
        :return:
        """
        with self.lock:
            self.displayed_dataset_key = event.new

class GemPyModule(Module):
    # child class of Model
    pass


class GeoMapModule:
    """

    """

    # TODO: When we move GeoMapModule import gempy just there

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

        self.geo_model = geo_model
        self.kinect_grid = grid
        self.geol_map = geol_map

        self.fault_line = self.create_fault_line(0, self.geo_model.geo_data_res.n_faults + 0.5001)
        self.main_contours = self.create_main_contours(self.kinect_grid.scale.extent[4],
                                                       self.kinect_grid.scale.extent[5])
        self.sub_contours = self.create_sub_contours(self.kinect_grid.scale.extent[4],
                                                     self.kinect_grid.scale.extent[5])

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
        sol = gp.compute_model_at(self.kinect_grid.depth_grid, self.geo_model)
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
            outfile:

        Returns:

        """

        self.geol_map.render_frame(block)

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


class ArucoMarkers:
    """
    class to detect Aruco markers in the kinect data (IR and RGB)
    An Area of interest can be specified, markers outside this area will be ignored
    TODO: run as loop in a thread, probably implement in API
    """

    def __init__(self, aruco_dict=None, Area=None):
        if not aruco_dict:
            self.aruco_dict = aruco.DICT_4X4_50  # set the default dictionary here
        else:
            self.aruco_dict = aruco_dict
        self.Area = Area  # set a square Area of interest here (Hot-Area)
        self.kinect = KinectV2()
        self.ir_markers = self.find_markers_ir(self.kinect)
        self.rgb_markers = self.find_markers_rgb(self.kinect)
        self.dict_markers_current = self.update_dict_markers_current()  # markers that were detected in the last frame
        #self.dict_markers_all =pd.DataFrame({}) # all markers ever detected with their last known position and timestamp
        self.dict_markers_all = self.dict_markers_current
        self.lock = threading.Lock  # thread lock object to avoid read-write collisions in multithreading.
        #self.trs_dst = self.change_point_RGB_to_DepthIR()
        self.ArucoImage = self.create_aruco_marker()


    def get_location_marker(self, corners):
        pr1 = int(numpy.mean(corners[:, 0]))
        pr2 = int(numpy.mean(corners[:, 1]))
        return pr1, pr2

    def aruco_detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(self.aruco_dict)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        return corners, ids, rejectedImgPoints

    def find_markers_ir(self, kinect: KinectV2):
        labels = {'ids', 'Corners_IR_x', 'Corners_IR_y'} #TODO: add orientation of aruco marker
        df = pd.DataFrame(columns=labels)
        list_values = df.set_index('ids')

        while len(list_values) < 4:

            minim = 0
            maxim = numpy.arange(1000, 30000, 500)
            IR = kinect.get_ir_frame_raw()
            for i in maxim:
                ir_use = numpy.interp(IR, (minim, i), (0, 255)).astype('uint8')
                ir3 = numpy.stack((ir_use, ir_use, ir_use), axis=2)
                corners, ids, rejectedImgPoints = self.aruco_detect(ir3)

                if not ids is None:
                    for j in range(len(ids)):
                        if ids[j] not in list_values.index.values:
                            x_loc, y_loc = self.get_location_marker(corners[j][0])
                            df_temp = pd.DataFrame({'ids': [ids[j][0]], 'Corners_IR_x': [x_loc], 'Corners_IR_y': [y_loc]})
                            df = pd.concat([df, df_temp], sort=False)
                            list_values = df.set_index('ids')

        self.ir_markers = list_values

        return self.ir_markers

    def find_markers_rgb(self, kinect :KinectV2):
        labels = {"ids", "Corners_RGB_x", "Corners_RGB_y"}  #TODO: add orientation of aruco marker
        df = pd.DataFrame(columns=labels)
        list_values_color = df.set_index("ids")

        while len(list_values_color) < 5:
            color = kinect.get_color()
            corners, ids, rejectedImgPoints = self.aruco_detect(color)

            if not ids is None:
                for j in range(len(ids)):
                    if ids[j] not in list_values_color.index.values:
                        x_loc, y_loc = self.get_location_marker(corners[j][0])
                        df_temp = pd.DataFrame({"ids": [ids[j][0]], "Corners_RGB_x": [x_loc], "Corners_RGB_y": [y_loc]})
                        df = pd.concat([df, df_temp], sort=False)
                        list_values_color = df.set_index("ids")

        self.rgb_markers = list_values_color

        return self.rgb_markers


    def update_dict_markers_current(self):

        ir_aruco_locations = self.ir_markers
        rgb_aruco_locations = self.rgb_markers
        self.dict_markers_current = pd.concat([ir_aruco_locations,rgb_aruco_locations], axis=1)
        return self.dict_markers_current

    def update_dict_markers_all(self):

        self.dict_markers_all.update(self.dict_markers_current)
        return self.dict_markers_all


    def erase_dict_markers_all(self):
        self.dict_markers_all = pd.DataFrame({})
        return self.dict_markers_all

    def change_point_RGB_to_DepthIR(self):
        """
        Get a perspective transform matrix to project points from RGB to Depth/IR space

        Args:
            src: location in x and y of the points from the source image (requires 4 points)
            dst: equivalence of position x and y from source image to destination image (requires 4 points)

        Returns:
            trs_dst: location in x and y of the projected point in Depth/IR space
        """
        full = self.dict_markers_current.dropna()
        mis = self.dict_markers_current[self.dict_markers_current.isna().any(1)]

        src = numpy.array(full[["Corners_RGB_x", "Corners_RGB_y"]]).astype(numpy.float32)
        dst = numpy.array(full[["Corners_IR_x", "Corners_IR_y"]]).astype(numpy.float32)

        trs_src = numpy.array([mis["Corners_RGB_x"], mis["Corners_RGB_y"], 1]).astype(numpy.float32)

        transform_perspective = cv2.getPerspectiveTransform(src, dst)

        trans_val = numpy.dot(transform_perspective, trs_src.T).astype("int")

        values = {"Corners_IR_x": trans_val[0], "Corners_IR_y": trans_val[1]}

        self.dict_markers_current = self.dict_markers_current.fillna(value=values)

        return self.dict_markers_current

    def create_aruco_marker(self, nx=1, ny=1,show=False):
        self.ArucoImage = 0
        if show is True:
            aruco_dictionary = aruco.Dictionary_get(self.aruco_dict)

            fig = plt.figure()
            for i in range(1, nx * ny + 1):
                ax = fig.add_subplot(ny, nx, i)
                img = aruco.drawMarker(aruco_dictionary, i, 2000)

                plt.imshow(img, cmap=plt.cm.gray, interpolation="nearest")
                ax.axis("off")

            plt.savefig("markers.pdf")
            plt.show()
            self.ArucoImage = img

        return self.ArucoImage

    def plot_ir_aruco_location(self, kinect : KinectV2):
        plt.figure(figsize=(20, 20))
        plt.imshow(kinect.get_ir_frame(), cmap="gray")
        plt.plot(self.dict_markers_current["Corners_IR_x"], self.dict_markers_current["Corners_IR_y"], "or")
        plt.show()
        self.ArucoImage = img
        return self.ArucoImage


    def plot_rgb_aruco_location(self, kinect: KinectV2):
        plt.figure(figsize=(20, 20))
        plt.imshow(kinect.get_color())
        plt.plot(self.dict_markers_current["Corners_RGB_x"], self.dict_markers_current["Corners_RGB_y"], "or")
        plt.show()