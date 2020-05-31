import warnings as warn
import numpy
import scipy

try:
    import freenect  # wrapper for KinectV1
except ImportError:
    print('Freenect module not found, KinectV1 will not work.')


class KinectV1:

    def __init__(self):
        # hard coded class attributes for KinectV1's native resolution
        self.name = 'kinect_v1'
        self.depth_width = 320
        self.depth_height = 240
        self.color_width = 640
        self.color_height = 480

        self.id = None
        self.device = None
        self.angle = None
        self.depth = None
        self.color = None
        self.setup()

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