import numpy
#from sandbox.calibration.calibration import CalibrationData


class Scale(object):
    """
    class that handles the scaling of whatever the sandbox shows and the real world sandbox
    self.extent: 3d extent of the model in the sandbox in model units.
    if no model extent is specified, the physical dimensions of the sandbox (x,y) and the set sensor range (z)
    are used.

    """

    def __init__(self, calibrationdata, xy_isometric=True, extent=None):
        """
        Args:
            calibrationdata:
            xy_isometric:
            extent:
        """
        if isinstance(calibrationdata, CalibrationData):
            self.calibration = calibrationdata
        else:
            raise TypeError("you must pass a valid calibration instance")

        self.xy_isometric = xy_isometric
        self.scale = [None, None, None]
        self.pixel_size = [None, None]
        self.pixel_scale = [None, None]

        if extent is None:  # extent should be array with shape (6,) or convert to list?
            self.extent = numpy.asarray([
                0.0,
                self.calibration.box_width,
                0.0,
                self.calibration.box_height,
                self.calibration.s_min,
                self.calibration.s_max,
            ])

        else:
            self.extent = numpy.asarray(extent)  # check: array with 6 entries!

    @property
    def output_res(self):
        # this is the dimension of the cropped kinect frame
        return self.calibration.s_frame_width, self.calibration.s_frame_height

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

        self.pixel_scale[0] = float(self.extent[1] - self.extent[0]) / float(self.output_res[0])
        self.pixel_scale[1] = float(self.extent[3] - self.extent[2]) / float(self.output_res[1])
        self.pixel_size[0] = float(self.calibration.box_width) / float(self.output_res[0])
        self.pixel_size[1] = float(self.calibration.box_height) / float(self.output_res[1])

        # TODO: change the extent in place!! or create a new extent object that stores the extent after that modification.
        if self.xy_isometric:  # model is extended in one horizontal direction to fit  into box while the scale
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
        self.scale[2] = float(self.extent[5] - self.extent[4]) / (self.calibration.s_max - self.calibration.s_min)
        print("scale in Model units/ mm (X,Y,Z): " + str(self.scale))

    # TODO: manually define zscale and either lower or upper limit of Z, adjust rest accordingly.
