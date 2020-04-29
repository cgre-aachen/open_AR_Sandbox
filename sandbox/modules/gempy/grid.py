import numpy
from .scale import Scale


class Grid(object):
    """
    class for grid objects. a grid stores the 3D coordinate of each pixel recorded by the kinect in model coordinates
    a calibration object must be provided, it is used to crop the kinect data to the area of interest
    TODO:  The cropping should be done in the kinect class, with calibration_data passed explicitly to the method! Do this for all the cases where calibration data is needed!
    """

    def __init__(self, calibration=None, scale=None):
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

    def create_empty_depth_grid(self):
        """
        Sets up XY grid (Z is empty, that is where the name is coming from)

        Returns:

        """
        width = numpy.linspace(self.scale.extent[0], self.scale.extent[1], self.scale.output_res[0])
        height = numpy.linspace(self.scale.extent[2], self.scale.extent[3], self.scale.output_res[1])
        xx, yy = numpy.meshgrid(width, height)
        self.empty_depth_grid = numpy.vstack([xx.ravel(), yy.ravel()]).T

        print("the shown extent is [" + str(self.empty_depth_grid[0, 0]) + ", " +
              str(self.empty_depth_grid[-1, 0]) + ", " +
              str(self.empty_depth_grid[0, 1]) + ", " +
              str(self.empty_depth_grid[-1, 1]) + "] "
              )

    def update_grid(self, cropped_frame):
        """
        The frame that is passed here is cropped and clipped
        Appends the z (depth) coordinate to the empty depth grid.
        this has to be done every frame while the xy coordinates only change if the calibration or model extent is changed.
        For performance reasons these steps are therefore separated.

        Args:
            cropped_frame: The frame that is passed here is cropped and clipped

        Returns:

        """
        scaled_frame = self.scale.extent[5] - \
                       ((cropped_frame - self.calibration.s_min) /
                        (self.calibration.s_max - self.calibration.s_min) *
                        (self.scale.extent[5] - self.scale.extent[4]))

        flattened_depth = scaled_frame.ravel()
        depth_grid = numpy.c_[self.empty_depth_grid, flattened_depth]

        self.depth_grid = depth_grid
