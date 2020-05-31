import numpy

try:
    from pykinect2 import PyKinectV2  # Wrapper for KinectV2 Windows SDK
    from pykinect2 import PyKinectRuntime
except ImportError:
    print('pykinect2 module not found, KinectV2 will not work.')


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
        self.depth = None
        self.color = None
        self.device = None
        self.setup()

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

        return self.color
