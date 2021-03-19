import numpy
import platform
import threading
from sandbox import set_logger
logger = set_logger(__name__)
_platform = platform.system()
try:
    if _platform == 'Windows':
        from pykinect2 import PyKinectV2  # Wrapper for KinectV2 Windows SDK
        from pykinect2 import PyKinectRuntime
    elif _platform == 'Linux':
        from freenect2 import Device, FrameType
except ImportError:
    logger.warning('dependencies not found for KinectV2 to work. Check installation and try again', exc_info=True)


class KinectV2:
    """
    control class for the KinectV2 based on the Python wrappers of the official Microsoft SDK
    Init the kinect and provides a method that returns the scanned depth image as numpy array.
    Also we do gaussian blurring to get smoother surfaces.

    """
    def __init__(self):
        # hard coded class attributes for KinectV2's native resolution
        self.name = 'kinect_v2'
        self.depth_width = 512
        self.depth_height = 424
        self.color_width = 1920
        self.color_height = 1080

        self._init_device()

        self.depth = self.get_frame()
        self.color = self.get_color()
        logger.info("KinectV2 initialized.")

    def _init_device(self):
        """
        creates the self.device parameter to start the stream of frames
        Returns:

        """
        if _platform == 'Windows':
            self.device = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color |
                                                          PyKinectV2.FrameSourceTypes_Depth |
                                                          PyKinectV2.FrameSourceTypes_Infrared)

        elif _platform == 'Linux':
            # Threading
            self._lock = threading.Lock()
            self._thread = None
            self._thread_status = 'stopped'  # status: 'stopped', 'running'

            self.device = Device()
            self._color = numpy.zeros((self.color_height, self.color_width, 4))
            self._depth = numpy.zeros((self.depth_height, self.depth_width))
            self._ir = numpy.zeros((self.depth_height, self.depth_width))
            self._run()
            logger.info("Searching first frame")
            while True:
                if not numpy.all(self._depth == 0):
                    logger.info("First frame found")
                    break

        else:
            logger.error(_platform + "not implemented")
            raise NotImplementedError

    def _run(self):
        """
        Run the thread when _platform is linux
        """
        if self._thread_status != 'running':
            self._thread_status = 'running'
            self._thread = threading.Thread(target=self._open_kinect_frame_stream, daemon=True, )
            self._thread.start()
            logger.info('Acquiring frames...')
        else:
            logger.info('Already running.')

    def _stop(self):
        """
        Stop the thread when _platform is linux
        """
        if self._thread_status is not 'stopped':
            self._thread_status = 'stopped'  # set flag to end thread loop
            self._thread.join()  # wait for the thread to finish
            logger.info('Stopping frame acquisition.')
        else:
            logger.info('thread was not running.')

    def _open_kinect_frame_stream(self):
        """
        keep the stream open to adquire new frames when using linux
        """
        frames = {}
        with self.device.running():
            for type_, frame in self.device:
                frames[type_] = frame
                if FrameType.Color in frames:
                    self._color = frames[FrameType.Color].to_array()
                if FrameType.Depth in frames:
                    self._depth = frames[FrameType.Depth].to_array()
                if FrameType.Ir in frames:
                    self._ir = frames[FrameType.Ir].to_array()
                if self._thread_status != "running":
                    break

    def get_frame(self):
        """
        Args:
        Returns:
               2D Array of the shape(424, 512) containing the depth information of the latest frame in mm
        """
        if _platform == 'Windows':
            depth_flattened = self.device.get_last_depth_frame()
            self.depth = depth_flattened.reshape(
                (self.depth_height, self.depth_width))  # reshape the array to 2D with native resolution of the kinectV2
        elif _platform == 'Linux':
            # assert self._thread_status == "running"
            self.depth = self._depth
        return self.depth

    def get_ir_frame_raw(self):
        """
        Args:
        Returns:
               2D Array of the shape(424, 512) containing the raw infrared intensity in (uint16) of the last frame
        """
        if _platform == 'Windows':
            ir_flattened = self.device.get_last_infrared_frame()
            # reshape the array to 2D with native resolution of the kinectV2
            self.ir_frame_raw = numpy.flipud(ir_flattened.reshape((self.depth_height, self.depth_width)))
        elif _platform == 'Linux':
            # assert self._thread_status == "running"
            self.ir_frame_raw = self._ir
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
        """

        Returns:

        """
        if _platform == 'Windows':
            color= numpy.array([self.device.get_last_color_frame()])

        elif _platform == 'Linux':
            # assert self._thread_status == "running"
            color = self._color

        resolution_camera = self.color_height * self.color_width  # resolution camera Kinect V2
        # Palette of colors in RGB / Cut of 4th column marked as intensity
        palette = numpy.reshape(color, (resolution_camera, 4))[:, [2, 1, 0]]
        position_palette = numpy.reshape(numpy.arange(0, len(palette), 1), (self.color_height, self.color_width))
        self.color = numpy.flipud(palette[position_palette])
        return self.color
