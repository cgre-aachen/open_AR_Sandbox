import logging
import numpy
import scipy
import scipy.ndimage
from warnings import warn

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
    def __init__(self, calibrationdata, name='kinect_v2', gauss_filter=True,
                 n_frames=3, gauss_sigma=3, **kwargs):

        # TODO this should be a dict
        self.calib = calibrationdata
        if name == 'kinect_v1':
            try:
                import freenct
                self.Sensor = KinectV1()
            except ImportError:
                raise ImportError('Kinect v1 dependencies are not installed')
        elif name == 'kinect_v2':
            self.Sensor = KinectV2()
        elif name == 'dummy':
            self.Sensor = DummySensor(**kwargs)
        else:
            warn("Unrecognized sensor name. Activating dummy sensor")
            self.Sensor = DummySensor(**kwargs)

        # filter parameters
        self.filter = gauss_filter
        self.n_frames = n_frames
        self.sigma_gauss = gauss_sigma

        self.calib.s_name = self.Sensor.name
        self.calib.s_width = self.Sensor.depth_width
        self.calib.s_width = self.Sensor.depth_width
        self.calib.s_height = self.Sensor.depth_height
        self.depth = None
        self.get_frame()

    def load_calibration(self):
        raise NotImplementedError

    # TODO Move properties

    def get_frame(self):
        """Grab a new height numpy array

        With the Dummy sensor it will sample noise
        """
        # collect last n frames in a stack
        depth_array = self.Sensor.get_frame()

        for i in range(self.n_frames - 1):
            depth_array = numpy.dstack([depth_array, self.Sensor.get_frame()])
        # calculate mean values ignoring zeros by masking them
        depth_array_masked = numpy.ma.masked_where(depth_array == 0, depth_array)  # needed for V2?
        self.depth = numpy.ma.mean(depth_array_masked, axis=2)
        if self.filter:
            # apply gaussian filter
            self.depth = scipy.ndimage.filters.gaussian_filter(self.depth, self.sigma_gauss)

        return self.depth
