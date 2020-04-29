from abc import ABCMeta
from abc import abstractmethod
import logging
import numpy
import scipy
from scipy.interpolate import griddata  # for DummySensor
import scipy.ndimage
from scipy.spatial.distance import cdist  # for DummySensor

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

# logging and exception handling
verbose = False
if verbose:
    logging.basicConfig(filename="main.log",
                        filemode='w',
                        level=logging.WARNING,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        )

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





