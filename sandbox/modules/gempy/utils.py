import numpy


def get_scale(physical_extent: list, model_extent: list, sensor_extent: list, xy_isometric: bool = False):
    """
    Calculates the factors for the coordinates transformation kinect-extent
    Model is extended in one horizontal direction to fit  into box while the scale
    pixel_scale [modelunits/pixel]: XY scaling factor
    pixel_size [mm/pixel]
    scale in model units
    Args:
        physical_extent: [box_width, box_height]
        model_extent: [x_origin, xmax, yorigin, ymax, zmin, zmax]
        sensor_extent:  [0, sensor_width, 0. sensor_height, vmin, vmax]
        xy_isometric: True
    Returns:
        scale in model units, pixel_scale [modelunits/pixel], pixel_size [mm/pixel]
    """
    scale = [None, None, None]
    pixel_size = [None, None]
    pixel_scale = [None, None]

    pixel_scale[0] = float(model_extent[1] - model_extent[0]) / float(sensor_extent[1])
    pixel_scale[1] = float(model_extent[3] - model_extent[2]) / float(sensor_extent[3])

    pixel_size[0] = float(physical_extent[0]) / float(sensor_extent[1])
    pixel_size[1] = float(physical_extent[1]) / float(sensor_extent[3])

    # TODO: change the extent in place!! or create a new extent object that stores the extent after that modification.
    if xy_isometric:  # model is extended in one horizontal direction to fit  into box while the scale
        # in both directions is maintained
        print("Aspect ratio of the model is fixed in XY")
        if pixel_scale[0] >= pixel_scale[1]:
            pixel_scale[1] = pixel_scale[0]
            print("Model size is limited by X dimension")
        else:
            pixel_scale[0] = pixel_scale[1]
            print("Model size is limited by Y dimension")

    scale[0] = pixel_scale[0] / pixel_size[0]
    scale[1] = pixel_scale[1] / pixel_size[1]
    # Vertical scaling
    scale[2] = float(model_extent[5] - model_extent[4]) / (sensor_extent[5] - sensor_extent[4])
    print("scale in Model units/ mm (X,Y,Z): " + str(scale))
    return scale, pixel_scale, pixel_size


class Grid(object):
    """
    class for grid objects. a grid stores the 3D coordinate of each pixel recorded by the kinect in model coordinates
    """

    def __init__(self, physical_extent: list, model_extent: list, sensor_extent: list, scale=None):
        """

        Args:
            physical_extent: [box_width, box_height]
            model_extent: [x_origin, xmax, yorigin, ymax, zmin, zmax]
            sensor_extent:  [0, sensor_width, 0. sensor_height, sensor_min, sensor_max]
         Returns:
            None
        """
        if scale is None:
            self.scale, _, _ = get_scale(physical_extent, model_extent, sensor_extent)
        else:
            self.scale = scale

        self.sensor_extent = sensor_extent
        self.physical_extent = physical_extent
        self.model_extent = model_extent

        self.depth_grid = None
        self.empty_depth_grid = None
        self.create_empty_depth_grid()

    def create_empty_depth_grid(self):
        """
        Sets up XY grid (Z is empty, that is where the name is coming from)
        Returns:

        """
        width = numpy.linspace(self.model_extent[0], self.model_extent[1], self.sensor_extent[1])
        height = numpy.linspace(self.model_extent[2], self.model_extent[3], self.sensor_extent[3])
        xx, yy = numpy.meshgrid(width, height)
        self.empty_depth_grid = numpy.vstack([xx.ravel(), yy.ravel()]).T

        print("the shown extent is [" + str(self.empty_depth_grid[0, 0]) + ", " +
              str(self.empty_depth_grid[-1, 0]) + ", " +
              str(self.empty_depth_grid[0, 1]) + ", " +
              str(self.empty_depth_grid[-1, 1]) + "] "
              )

    def scale_frame(self, frame):
        """Method to scale the frame"""
        if self.model_extent[-2] < 0:
            displace = self.model_extent[-2] * (-1) * (self.sensor_extent[-1] - self.sensor_extent[-2]) / (
                    self.model_extent[-1] - self.model_extent[-2])
            scaled_frame = frame - displace
            # now we set 2 regions. One above sea level and one below sea level. So now we can normalize these two
            # regions above 0
            scaled_frame[scaled_frame > 0] = scaled_frame[scaled_frame > 0] * (self.model_extent[-1] /
                                                                               (self.sensor_extent[-1]-displace))
            # below 0
            scaled_frame[scaled_frame < 0] = scaled_frame[scaled_frame < 0] * (self.model_extent[-2] /
                                                                               (self.sensor_extent[-2]-displace))
        else:
            scaled_frame = frame * self.scale[2]
            scaled_frame = scaled_frame + self.model_extent[-2]
        return scaled_frame

    def update_grid(self, scale_frame):
        """
        The frame that is passed here is cropped and clipped
        Appends the z (depth) coordinate to the empty depth grid.
        this has to be done every frame while the xy coordinates
        only change if the calibration or model extent is changed.
        For performance reasons these steps are therefore separated.

        Args:
            scale_frame: The frame that is passed here is cropped and clipped

        Returns:
        """
        flattened_depth = scale_frame.ravel()
        depth_grid = numpy.c_[self.empty_depth_grid, flattened_depth]

        self.depth_grid = depth_grid
