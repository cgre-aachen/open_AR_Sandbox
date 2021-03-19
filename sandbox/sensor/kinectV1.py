import numpy
from sandbox import set_logger
logger = set_logger(__name__)

try:
    import freenect  # wrapper for KinectV1
except ImportError:
    logger.warning('Freenect module not found, KinectV1 will not work', exc_info=True)


class KinectV1:

    def __init__(self):
        # hard coded class attributes for KinectV1's native resolution
        self.name = 'kinect_v1'
        self.depth_width = 320
        self.depth_height = 240
        self.color_width = 640
        self.color_height = 480

        self.id = 0
        self.device = None
        self.depth = None
        self.color = None
        logger.warning('Two kernels cannot access the Kinect at the same time. '
                       'This will lead to a sudden death of the kernel. '
                       'Be sure no other kernel is running before you initialize a KinectV1 object.')

        logger.info("looking for kinect...")
        ctx = freenect.init()
        self.device = freenect.open_device(ctx, self.id)
        print(self.id)
        freenect.close_device(self.device)  # TODO Test if this has to be done!
        # get the first Depth frame already (the first one takes much longer than the following)
        self.depth = self.get_frame()
        logger.info("KinectV1 initialized.")

    def get_frame(self):
        self.depth = freenect.sync_get_depth(index=self.id, format=freenect.DEPTH_MM)[0]
        self.depth = numpy.fliplr(self.depth)
        return self.depth

    def get_color(self):
        """
        Returns:

        """
        self.color = freenect.sync_get_video(index=self.id)[0]
        self.color = numpy.fliplr(self.color)
        return self.color
