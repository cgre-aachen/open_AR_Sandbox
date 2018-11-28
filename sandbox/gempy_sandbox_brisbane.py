#sys.path.append('/home/miguel/PycharmProjects/gempy')
import weakref
import numpy
import scipy
import gempy
import matplotlib.pyplot as plt
from itertools import count
from sandbox.Sandbox import Calibration

#TODO: make calibrations a mandatory argument in ALL Classes!!!


class Scale:
    def __init__(self, calibration=None, xy_isometric=True, extent=None):
        if isinstance(calibration, Calibration):
            self.calibration = calibration
        else:
            raise TypeError("you must pass a valid calibration instance")
        self.xy_isometric = xy_isometric
        self.scale = [None, None, None]
        self.pixel_size = [None, None]
        self.output_res = None

        if extent is None:  # extent should be array with shape (6,) or convert to list?
            self.extent = numpy.asarray([
                self.calibration.calibration_data['x_lim'][0],
                self.calibration.calibration_data['x_lim'][1],
                self.calibration.calibration_data['y_lim'][0],
                self.calibration.calibration_data['y_lim'][1],
                self.calibration.calibration_data['z_range'][0],
                self.calibration.calibration_data['x_range'][1],
                ])

        else:
            self.extent = numpy.asarray(extent)  # check: array with 6 entries!

    def calculate_scales(self):
        """
        calculates the factors for the coordinates transformation kinect-extent
        :return:
        """
        self.output_res = (self.calibration.calibration_data['x_lim'][1] -
                           self.calibration.calibration_data['x_lim'][0],
                           self.calibration.calibration_data['y_lim'][1] -
                           self.calibration.calibration_data['y_lim'][0])
        self.pixel_size[0] = float(self.extent[1] - self.extent[0]) / float(self.output_res[0])
        self.pixel_size[1] = float(self.extent[3] - self.extent[2]) / float(self.output_res[1])

        if self.xy_isometric == True:  # model is scaled to fit into box
            print("Aspect ratio of the model is fixed in XY")
            if self.pixel_size[0] >= self.pixel_size[1]:
                self.pixel_size[1] = self.pixel_size[0]
                print("Model size is limited by X dimension")
            else:
                self.pixel_size[0] = self.pixel_size[1]
                print("Model size is limited by Y dimension")

        self.scale[0] = self.pixel_size[0]
        self.scale[1] = self.pixel_size[1]
        self.scale[2] = float(self.extent[5] - self.extent[4]) / (
                self.calibration.calibration_data['z_range'][1] -
                self.calibration.calibration_data['z_range'][0])
        print("scale in Model units/ mm (X,Y,Z): " + str(self.scale))

    # TODO: manually define zscale and either lower or upper limit of Z, adjust rest accordingly.


class Grid:
    def __init__(self, calibration=None, scale=None,):
        if isinstance(calibration, Calibration):
            self.calibration = calibration
        else:
            raise TypeError("you must pass a valid calibration instance")
        if isinstance(scale, Scale):
            self.scale= scale
        else:
            self.scale = Scale(calibration=self.calibration)
            print("no scale provided or scale invalid. A default scale instance is used")
        self.empty_depth_grid=None
        self.create_empty_depth_grid()

    def create_empty_depth_grid(self):
        """
        sets up XY grid (Z is empty that is the name coming from)
        :return:
        """
        grid_list = []
        self.output_res = (self.calibration.calibration_data['x_lim'][1] -
                           self.calibration.calibration_data['x_lim'][0],
                           self.calibration.calibration_data['y_lim'][1] -
                           self.calibration.calibration_data['y_lim'][0])
        for x in range(self.output_res[1]):
            for y in range(self.output_res[0]):
                grid_list.append([y * self.pixel_size[1] + self.extent[2], x * self.pixel_size[0] + self.extent[0]])

        empty_depth_grid = numpy.array(grid_list)
        self.empty_depth_grid = empty_depth_grid

        # return self.empty_depth_grid

    def update_grid(self, depth):
        depth = numpy.fliplr(depth)  ##dirty workaround to get the it running with new gempy version.
        filtered_depth = numpy.ma.masked_outside(depth, self.calibration.calibration_data['z_range'][0],
                                                 self.calibration.calibration_data['z_range'][1])
        scaled_depth = self.extent[5] - (
                (filtered_depth - self.calibration.calibration_data['z_range'][0]) / (
                self.calibration.calibration_data['z_range'][1] -
                self.calibration.calibration_data['z_range'][0]) * (self.extent[5] - self.extent[4]))
        rotated_depth = scipy.ndimage.rotate(scaled_depth, self.calibration.calibration_data['rot_angle'],
                                             reshape=False)
        cropped_depth = rotated_depth[self.calibration.calibration_data['y_lim'][0]:
                                      self.calibration.calibration_data['y_lim'][1],
                        self.calibration.calibration_data['x_lim'][0]:
                        self.calibration.calibration_data['x_lim'][1]]

        flattened_depth = numpy.reshape(cropped_depth, (numpy.shape(self.empty_depth_grid)[0], 1))
        depth_grid = numpy.concatenate((self.empty_depth_grid, flattened_depth), axis=1)

        self.depth_grid = depth_grid

class Contour:
    def __init__(self, x,y,z, start, end,step, show=True, show_labels=False, linewidth=1.0, colors=[(0, 0, 0, 1.0)]):
        self.x = x
        self.y = y
        self.z = z
        self.start = start
        self.end = end
        self.step = step
        self.show = show
        self.show_labels = show_labels
        self.linewidth = linewidth
        self.colors = colors
        self.contours = None


    def add_contours:
        if self.show is True:
            self.contours=plt.contour(self.x, self.y, sewlf.z, levels=self.sub_contours, linewidths=self.linewidth, colors=self.colors)
            if self.show_labels is True:
                plt.clabel(self.contours,inline=0, fontsize=15, fmt='%3.0f')


class Plot:
    """
    handles the plotting of a sandbox model
    add contours by providing a list of contour objects
    """
    def __init__(self, rasterdata=None, calibration=None, cmap=None, norm=None, lot=None, contours=None, outfile=None):
        if isinstance(calibration, Calibration):
            self.calibration = calibration
        else:
            raise TypeError("you must pass a valid calibration instance")
        self.rasterdata=rasterdata
        self.cmap = cmap
        self.norm = norm
        self.lot = lot
        self.contours=contours
        self.output_res = (
            self.calibration.calibration_data['x_lim'][1] -
            self.calibration.calibration_data['x_lim'][0],
            self.calibration.calibration_data['y_lim'][1] -
            self.calibration.calibration_data['y_lim'][0]
                           )
        self.block = self.rasterdata.reshape((self.output_res[1], self.output_res[0]))
        self.h = self.calibration.calibration_data['scale_factor'] * (self.output_res[1]) / 100.0
        self.w = self.calibration.calibration_data['scale_factor'] * (self.output_res[0]) / 100.0

        self.fig = plt.figure(figsize=(w, h), dpi=100, frameon=False)
        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        self.ax.set_axis_off()
        self.fig.add_axes(self.ax)
        self.outfile = outfile

    def render_frame(self):
        self.ax.pcolormesh(self.block, cmap=self.cmap, norm=self.norm)

        for contour in self.contours:
            contour.add_contours()


        if self.outfile is None:
            plt.show()
            plt.close()
        else:
            plt.savefig(self.outfile, pad_inches=0)
            plt.close(self.fig)


    def create_legend(self):
        # ...
        pass



### Still to refactor:

#TODO: use Descriptors
class Model:
    _ids = count(0)
    _instances = []

    def __init__(self, model, calibration=None, lock=None):
        self.id = next(self._ids)
        self.__class__._instances.append(weakref.proxy(self))

        self.legend = True
        self.model = model
        gempy.compute_model(self.model)
        self.empty_depth_grid = None
        self.depth_grid = None

        self.stop_threat = False
        self.lock = lock

        if calibration is None:
            try:
                self.calibration = Calibration._instances[-1]
                print("no calibration specified, using last calibration instance created: ",self.calibration)
            except:
                print("ERROR: no calibration instance found. please create a calibration")
                # parameters from the model:
        else:
            self.calibration = calibration


    def setup(self, start_stream=False):
        if start_stream == True:
            self.calibration.associated_projector.start_stream()
        self.calculate_scales()
        self.create_empty_depth_grid()

    def run(self):
        run_model(self)


## global functions to run the model in loop.
def run_model(model, calibration=None, kinect=None, projector=None, filter_depth=True, n_frames=5,
              sigma_gauss=4):  # continous run functions with exit handling
    if calibration == None:
        calibration = model.calibration
    if kinect == None:
        kinect = calibration.associated_kinect
    if projector == None:
        projector = calibration.associated_projector

    while True:
        if filter_depth == True:
            depth = kinect.get_filtered_frame(n_frames=n_frames, sigma_gauss=sigma_gauss)
        else:
            depth = kinect.get_frame()

        model.update_grid(depth)
        model.render_frame(depth, outfile="current_frame.jpeg")
        #time.sleep(self.delay)
        projector.show(input="current_frame.jpeg", rescale=False)

        if model.stop_threat is True:
            raise Exception('Threat stopped')
