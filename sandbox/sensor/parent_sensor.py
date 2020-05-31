from abc import ABCMeta
from abc import abstractmethod

import numpy
import scipy

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

