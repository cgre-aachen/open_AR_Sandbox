import logging
import numpy
import scipy
import scipy.ndimage
from warnings import warn
import json

from .kinectV1 import KinectV1
from .kinectV2 import KinectV2
from .dummy import DummySensor


# logging and exception handling
verbose = False
if verbose:
    logging.basicConfig(filename="main.log",
                        filemode='w',
                        level=logging.WARNING,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        )


class Sensor:
    """
    Wrapping API-class
    """
    def __init__(self, calibsensor: str = None, name: str ='kinect_v2', crop_values: bool = True,
                 clip_values: bool = False, gauss_filter: bool = True,
                 n_frames: int = 3, gauss_sigma: int = 3, **kwargs):

        self.version = '2.0.s'
        if calibsensor is None:
            self.s_name = name
            self.s_top = 10
            self.s_right = 10
            self.s_bottom = 10
            self.s_left = 10
            self.s_min = 700
            self.s_max = 1500
            self.box_width = 1000
            self.box_height = 800
        else:
            self.load_json(calibsensor)

        if name == 'kinect_v1':
            try:
                import freenct
                self.Sensor = KinectV1()
            except ImportError:
                raise ImportError('Kinect v1 dependencies are not installed')
        elif name == 'kinect_v2':
            try:
                import pykinect2
                self.Sensor = KinectV2()
            except ImportError:
                raise ImportError('Kinect v2 dependencies are not installed')
        elif name == 'dummy':
            self.Sensor = DummySensor(**kwargs)
        else:
            warn("Unrecognized sensor name. Activating dummy sensor")
            self.Sensor = DummySensor(**kwargs)

        # filter parameters
        self.filter = gauss_filter
        self.n_frames = n_frames
        self.sigma_gauss = gauss_sigma

        self.s_name = self.Sensor.name
        self.s_width = self.Sensor.depth_width
        self.s_width = self.Sensor.depth_width
        self.s_height = self.Sensor.depth_height
        self.depth = None
        self.crop = crop_values
        self.clip = clip_values
        self.get_frame()

    def get_raw_frame(self, gauss_filter: bool = True) -> numpy.ndarray:
        """Grab a new height numpy array

        With the Dummy sensor it will sample noise
        """
        # collect last n frames in a stack
        depth_array = self.Sensor.get_frame()
        for i in range(self.n_frames - 1):
            depth_array = numpy.dstack([depth_array, self.Sensor.get_frame()])
        # calculate mean values ignoring zeros by masking them
        depth_array_masked = numpy.ma.masked_where(depth_array == 0, depth_array)  # needed for V2?
        depth = numpy.ma.mean(depth_array_masked, axis=2)
        if gauss_filter:
            # apply gaussian filter
            depth = scipy.ndimage.filters.gaussian_filter(depth, self.sigma_gauss)

        return depth

    # computed parameters for easy access
    @property
    def s_frame_width(self): return self.s_width - self.s_left - self.s_right

    @property
    def s_frame_height(self): return self.s_height - self.s_top - self.s_bottom

    def load_json(self, file: str):
        """
         Load a calibration file (.JSON format) and actualizes the panel parameters
         Args:
             file: address of the calibration to load

         Returns:

         """
        with open(file) as calibration_json:
            data = json.load(calibration_json)
            if data['version'] == self.version:
                self.s_name = data['s_name']
                self.s_top = data['s_top']
                self.s_right = data['s_right']
                self.s_bottom = data['s_bottom']
                self.s_left = data['s_left']
                self.s_min = data['s_min']
                self.s_max = data['s_max']
                self.box_width = data['box_width']
                self.box_height = data['box_height']

                print("JSON configuration loaded for sensor.")
            else:
                print(
                    "JSON configuration incompatible.\nPlease select a valid calibration file or start a new calibration!")

    def save_json(self, file: str = 'sensor_calibration.json'):
        """
        Saves the current state of the sensor in a .JSON calibration file
        Args:
            file: address to save the calibration

        Returns:

        """
        with open(file, "w") as calibration_json:
            data = {"version": self.version,
                    "s_name": self.s_name,
                    "s_top": self.s_top,
                    "s_right": self.s_right,
                    "s_bottom": self.s_bottom,
                    "s_left": self.s_left,
                    "s_frame_width": self.s_frame_width,
                    "s_frame_height": self.s_frame_height,
                    "s_min": self.s_min,
                    "s_max": self.s_max,
                    "box_width": self.box_width,
                    "box_height": self.box_height}
            json.dump(data, calibration_json)
        print('JSON configuration file saved:', str(file))

    def crop_frame(self, frame: numpy.ndarray) -> numpy.ndarray:
        """ Crops the data frame according to the horizontal margins set up in the calibration
        """

        # TODO: Does not work yet for s_top = 0 and s_right = 0, which currently returns an empty frame!
        # TODO: Workaround: do not allow zeroes in calibration widget and use default value = 1
        # TODO: File numpy issue?
        crop = frame[self.s_bottom:-self.s_top, self.s_left:-self.s_right]
        return crop

    def crop_frame_workaround(self, frame: numpy.ndarray) -> numpy.ndarray:
        # bullet proof working example
        if self.s_top == 0 and self.s_right == 0:
            crop = frame[self.s_bottom:, self.s_left:]
        elif self.s_top == 0:
            crop = frame[self.s_bottom:, self.s_left:-self.s_right]
        elif self.s_right == 0:
            crop = frame[self.s_bottom:-self.s_top, self.s_left:]
        else:
            crop = frame[self.s_bottom:-self.s_top, self.s_left:-self.s_right]

        return crop

    def depth_mask(self, frame: numpy.ndarray) -> numpy.ndarray:
        """ Creates a boolean mask with True for all values within the set sensor range and False for every pixel
        above and below. If you also want to use clipping, make sure to use the mask before.
        """

        mask = numpy.ma.getmask(numpy.ma.masked_outside(frame, self.s_min, self.s_max))
        return mask

    def clip_frame(self, frame: numpy.ndarray) -> numpy.ndarray:
        """ Clips all values outside of the sensor range to the set s_min and s_max values.
        If you want to create a mask make sure to call depth_mask before performing the clip.
        """

        clip = numpy.clip(frame, self.s_min, self.s_max)
        return clip

    def get_frame(self) -> numpy.ndarray:
        frame = self.get_raw_frame(self.filter)
        if self.crop:
            frame = self.crop_frame(frame)
        if self.clip:
            frame = self.depth_mask(frame)
            frame = self.clip_frame(frame)
        self.depth = frame
        return self.depth

    @property
    def extent(self):
        """returns the extent in pixels used for the modules to indicate the dimensions of the plot in the sandbox"""
        return [0, self.s_frame_width, 0, self.s_frame_height, self.s_min, self.s_max]