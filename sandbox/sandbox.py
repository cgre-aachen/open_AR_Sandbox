import os
from warnings import warn
try:
    import freenect
    warn('Two kernels cannot access the kinect at the same time. This will lead to a sudden death of the kernel. ' \
         'Be sure no other kernel is running before initialize a kinect object.', RuntimeWarning)
except ImportError:
    warn('Freenect is not installed. if you are using the Kinect Version 2 on a windows machine, use the KinectV2 class!')

try:
    from pykinect2 import PyKinectV2 #try to import Wrapper for KinectV2 Windows SDK
    from pykinect2 import PyKinectRuntime

except ImportError:
    pass

try:
    import cv2

except ImportError:
   # warn('opencv is not installed. Object detection will not work')
    pass

import webbrowser
import pickle
import numpy
import scipy
import scipy.ndimage

from itertools import count
from PIL import Image, ImageDraw
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib
#import gempy.hackathon as hackathon
import IPython
import threading

# TODO: When we move GeoMapModule import gempy just there
import gempy as gp

class Kinect:  # add dummy
    """
    Masterclass for initializing the kinect.
    Init the kinect and provides a method that returns the scanned depth image as numpy array. Also we do the gaussian
    blurring to get smoother lines.
    """

    version_kinect = int(input("Version of Kinect using (1 or 2):"))
    def __init__(self,version_kinect=version_kinect, dummy=False, mirror=True):
        """
        Args:
            dummy:
            mirror:

        Returns:
        """
        self.resolution = (640, 480)  #TODO: check if this is used anywhere: this is the resolution of the camera! The depth image resolution is 320x240
        self.dummy = dummy
        self.mirror = mirror # TODO: check if this is used anywhere, then delete
        self.rgb_frame = None
        self.angle = None

        #TODO: include filter self.-filter parameters as function defaults
        self.n_frames = 3 #filter parameters
        self.sigma_gauss = 3
        self.filter = 'gaussian' #TODO: deprecate get_filtered_frame, make it switchable in runtime

        if version_kinect == 1:
            if self.dummy is False:
                print("looking for kinect...")
                self.ctx = freenect.init()
                self.dev = freenect.open_device(self.ctx, self.id)
                print(self.id)
                freenect.close_device(self.dev)  # TODO Test if this has to be done!

                self.depth = freenect.sync_get_depth(index=self.id, format=freenect.DEPTH_MM)[0]  # get the first Depth frame already (the first one takes much longer than the following)
                self.filtered_depth = None
                print("kinect initialized")
            else:
                self.filtered_depth = None
                self.depth = self.get_frame()
                print("dummy mode. get_frame() will return a synthetic depth frame, other functions may not work")

        elif version_kinect == 2:
            self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Infrared)
            self.depth = self.get_frame()
            self.color = self.get_color()
            self.mapped = self.get_mapped(self.depth)
            self.ir_frame_raw = self.get_ir_frame_raw()
            self.ir_frame = self.get_ir_frame()


        else:
            print("Choose a valid version for the Kinect (1 or 2). Please restart kernel")

class KinectV1 (Kinect):
    def set_angle(self, angle): #TODO: throw out
        """
        Args:
            angle:

        Returns:
            None
        """
        self.angle = angle
        freenect.set_tilt_degs(self.dev, self.angle)

    def get_frame(self, horizontal_slice=None):
        """
        Args:
            horizontal_slice:

        Returns:
        """
        if self.dummy is False:
            self.depth = freenect.sync_get_depth(index=self.id, format=freenect.DEPTH_MM)[0]
            self.depth = numpy.fliplr(self.depth)
            return self.depth
        else:
            synth_depth = numpy.zeros((480, 640))
            for x in range(640):
                for y in range(480):
                    if horizontal_slice is None:
                        synth_depth[y, x] = int(800 + 200 * (numpy.sin(2 * numpy.pi * x / 320)))
                    else:
                        synth_depth[y, x] = horizontal_slice
            self.depth = synth_depth
            return self.depth

    def get_filtered_frame(self, n_frames=None, sigma_gauss=None):
        """
        Args:
            n_frames:
            sigma_gauss:

        Returns:
        """
        if n_frames is None:
            n_frames = self.n_frames
        if sigma_gauss is None:
            sigma_gauss = self.sigma_gauss

        if self.dummy is True:
            self.get_frame()
            return self.depth
        elif self.filter == 'gaussian':

            depth_array = freenect.sync_get_depth(index=self.id, format=freenect.DEPTH_MM)[0]
            for i in range(n_frames - 1):
                depth_array = numpy.dstack([depth_array, freenect.sync_get_depth(index=self.id, format=freenect.DEPTH_MM)[0]])
            depth_array_masked = numpy.ma.masked_where(depth_array == 0, depth_array)
            self.depth = numpy.ma.mean(depth_array_masked, axis=2)
            self.depth = scipy.ndimage.filters.gaussian_filter(self.depth, sigma_gauss)
            return self.depth

    def get_rgb_frame(self):  # TODO: check if this can be thrown out
        """

        Returns:

        """
        if self.dummy is False:
            self.rgb_frame = freenect.sync_get_video(index=self.id)[0]
            self.rgb_frame = numpy.fliplr(self.rgb_frame)

            return self.rgb_frame
        else:
            pass

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


class KinectV2(Kinect):
    """
    control class for the KinectV2 based on the Python wrappers of the official Microsoft SDK
    Init the kinect and provides a method that returns the scanned depth image as numpy array.
    Also we do gaussian blurring to get smoother surfaces.

    """

    def get_frame(self):
        """

        Args:

        Returns:
               2D Array of the shape(424, 512) containing the depth information of the latest frame in mm

        """

        depth_flattened = self.kinect.get_last_depth_frame()
        self.depth = depth_flattened.reshape((424, 512)) #reshape the array to 2D with native resolution of the kinectV2
        return self.depth

    def get_ir_frame_raw(self):
        """

        Args:

        Returns:
               2D Array of the shape(424, 512) containing the raw infrared intensity in (uint16) of the last frame

        """
        ir_flattened = self.kinect.get_last_infrared_frame()
        self.ir_frame_raw = ir_flattened.reshape((424, 512)) #reshape the array to 2D with native resolution of the kinectV2
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

    def get_filtered_frame(self, n_frames=3, sigma_gauss=3):
        """

        Args:


        Returns:
            2D Array of the shape(424, 512) containing the depth information of the latest frame in mm after stacking of
             self.n_frames and gaussian blurring with a kernel of self.sigma_gauss pixels.
        """
        if self.filter == 'gaussian':

            depth_array = self.get_frame()
            for i in range(n_frames - 1):
                depth_array_masked = numpy.dstack([depth_array, self.get_frame()])
            self.depth = numpy.ma.mean(depth_array_masked, axis=2)
            self.depth = scipy.ndimage.filters.gaussian_filter(self.depth, self.sigma_gauss)
            return self.depth

    def get_color(self):
        color_flattened = self.kinect.get_last_color_frame()
        resolution_camera = 1920 * 1080 # resolution camera Kinect V2
        # Palette of colors in RGB / Cut of 4th column marked as intensity
        palette = numpy.reshape(numpy.array([color_flattened]), (resolution_camera, 4))[:,[2,1,0]]
        position_palette = numpy.reshape(numpy.arange(0, len(palette), 1), (1080, 1920))
        self.color = numpy.flipud(palette[position_palette])
        return self.color

    def get_mapped(self, position):
        """

        Args:

        Returns:

        """
        map = self.kinect.body_joints_to_color_space(self, position)
        self.mapped = map
        return self.mapped


class Calibration:
    """
    TODO:refactor completely! Make clear distinction between the calibration methods and calibration Data!
    Tune calibration parameters. Save calibration file. Have methods to project so we can see what we are calibrating
    """

    def __init__(self, associated_projector=None, associated_kinect=None, calibration_file=None):
        """

        Args:
            associated_projector:
            associated_kinect:
            calibration_file:
        """
        self.associated_projector = associated_projector
        self.projector_resolution = associated_projector.resolution
        self.associated_kinect = associated_kinect
        if calibration_file is None:
            self.calibration_file = "calibration" + str(self.id) + ".dat"

        self.calibration_data = CalibrationData(
             legend_x_lim=(self.projector_resolution[1] - 50, self.projector_resolution[0] - 1),
             legend_y_lim=(self.projector_resolution[1] - 100, self.projector_resolution[1] - 50),
             profile_area=False,
             profile_x_lim=(self.projector_resolution[0] - 50, self.projector_resolution[0] - 1),
             profile_y_lim=(self.projector_resolution[1] - 100, self.projector_resolution[1] - 1),
             hot_area=False,
             hot_x_lim=(self.projector_resolution[0] - 50, self.projector_resolution[0] - 1),
             hot_y_lim=(self.projector_resolution[1] - 100, self.projector_resolution[1] - 1)
                             )
        self.cmap = None
        self.contours = True
        self.n_contours = 20
        self.contour_levels = numpy.arange(self.calibration_data.z_range[0],
                                           self.calibration_data.z_range[1],
                                           float(self.calibration_data.z_range[1] - self.calibration_data.z_range[0]) / self.n_contours)

    # ...

    def load(self, calibration_file=None):
        """

        Args:
            calibration_file:

        Returns:

        """
        if calibration_file == None:
            calibration_file = self.calibration_file
        try:
            self.calibration_data = pickle.load(open(calibration_file, 'rb'))
            if not isinstance(self.calibration_data, CalibrationData):
                raise TypeError("loaded data is not a Calibration File object")
        except OSError:
            print("calibration data file not found. Using default values")

    def save(self, calibration_file=None):
        """

        Args:
            calibration_file:

        Returns:

        """
        if calibration_file is None:
            calibration_file = self.calibration_file
        pickle.dump(self.calibration_data, open(calibration_file, 'wb'))
        print("calibration saved to " + str(calibration_file))

    def create(self):
        """

        Returns:

        """
        if self.associated_projector is None:
                print("Error: no Projector instance found.")

        if self.associated_kinect is None:
                print("Error: no kinect instance found.")

        def calibrate(rot_angle, x_lim, y_lim, x_pos, y_pos, scale_factor, z_range, box_width, box_height, legend_area,
                      legend_x_lim, legend_y_lim, profile_area, profile_x_lim, profile_y_lim, hot_area, hot_x_lim,
                      hot_y_lim, close_click):
            """

            Args:
                rot_angle:
                x_lim:
                y_lim:
                x_pos:
                y_pos:
                scale_factor:
                z_range:
                box_width:
                box_height:
                legend_area:
                legend_x_lim:
                legend_y_lim:
                profile_area:
                profile_x_lim:
                profile_y_lim:
                hot_area:
                hot_x_lim:
                hot_y_lim:
                close_click:

            Returns:

            """
            depth = self.associated_kinect.get_frame()
            depth_rotated = scipy.ndimage.rotate(depth, rot_angle, reshape=False)
            depth_cropped = depth_rotated[y_lim[0]:y_lim[1], x_lim[0]:x_lim[1]]
            depth_masked = numpy.ma.masked_outside(depth_cropped, self.calibration_data.z_range[0],
                                                   self.calibration_data.z_range[1])  # depth pixels outside of range are white, no data pixe;ls are black.

            self.cmap = matplotlib.colors.Colormap('viridis')
            self.cmap.set_bad('white', 800)
            plt.set_cmap(self.cmap)
            h = (y_lim[1] - y_lim[0]) / 100.0
            w = (x_lim[1] - x_lim[0]) / 100.0

            fig = plt.figure(figsize=(w, h), dpi=100, frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.pcolormesh(depth_masked, vmin=self.calibration_data.z_range[0],
                          vmax=self.calibration_data.z_range[1])

            if self.contours is True: # draw contours
                self.contour_levels = numpy.arange(self.calibration_data.z_range[0],
                                                   self.calibration_data.z_range[1],
                                                   float(self.calibration_data.z_range[1] -
                                                         self.calibration_data.z_range[
                                                             0]) / self.n_contours) # update contour levels
                plt.contour(depth_masked,levels=self.contour_levels, linewidths=1.0, colors=[(0, 0, 0, 1.0)])

            plt.savefig(os.path.join(self.associated_projector.work_directory,'current_frame.png'), pad_inches=0)
            plt.close(fig)

            self.calibration_data = CalibrationData(
                                    rot_angle=rot_angle,
                                    x_lim=x_lim,
                                    y_lim=y_lim,
                                    x_pos=x_pos,
                                    y_pos=y_pos,
                                    scale_factor=scale_factor,
                                    z_range=z_range,
                                    box_width=box_width,
                                    box_height=box_height,
                                    legend_area=legend_area,
                                    legend_x_lim=legend_x_lim,
                                    legend_y_lim=legend_y_lim,
                                    profile_area=profile_area,
                                    profile_x_lim=profile_x_lim,
                                    profile_y_lim=profile_y_lim,
                                    hot_area=hot_area,
                                    hot_x_lim=hot_x_lim,
                                    hot_y_lim=hot_y_lim
                                  )

            if self.calibration_data.legend_area is not False:
                legend = Image.new('RGB', (
                self.calibration_data.legend_x_lim[1] - self.calibration_data.legend_x_lim[0],
                self.calibration_data.legend_y_lim[1] - self.calibration_data.legend_y_lim[0]), color='white')
                ImageDraw.Draw(legend).text((10, 10), "Legend", fill=(255, 255, 0))
                legend.save(os.path.join(self.associated_projector.work_directory,'legend.png'))
            if self.calibration_data.profile_area is not False:
                profile = Image.new('RGB', (
                self.calibration_data.profile_x_lim[1] - self.calibration_data.profile_x_lim[0],
                self.calibration_data.profile_y_lim[1] - self.calibration_data.profile_y_lim[0]), color='blue')
                ImageDraw.Draw(profile).text((10, 10), "Profile", fill=(255, 255, 0))
                profile.save(os.path.join(self.associated_projector.work_directory,'profile.png'))
            if self.calibration_data.hot_area is not False:
                hot = Image.new('RGB', (self.calibration_data.hot_x_lim[1] - self.calibration_data.hot_x_lim[0],
                                        self.calibration_data.hot_y_lim[1] - self.calibration_data.hot_y_lim[0]),
                                color='red')
                ImageDraw.Draw(hot).text((10, 10), "Hot Area", fill=(255, 255, 0))
                hot.save(os.path.join(self.associated_projector.work_directory,'hot.png'))
            self.associated_projector.show()
            if close_click == True:
                calibration_widget.close()

        calibration_widget = widgets.interactive(calibrate,
                                                 rot_angle=widgets.IntSlider(
                                                     value=self.calibration_data.rot_angle, min=-180, max=180,
                                                     step=1, continuous_update=False),
                                                 x_lim=widgets.IntRangeSlider(
                                                     value=[self.calibration_data.x_lim[0],
                                                            self.calibration_data.x_lim[1]],
                                                     min=0, max=640, step=1, continuous_update=False),
                                                 y_lim=widgets.IntRangeSlider(
                                                     value=[self.calibration_data.y_lim[0],
                                                            self.calibration_data.y_lim[1]],
                                                     min=0, max=480, step=1, continuous_update=False),
                                                 x_pos=widgets.IntSlider(value=self.calibration_data.x_pos, min=0,
                                                                         max=self.projector_resolution[0]),
                                                 y_pos=widgets.IntSlider(value=self.calibration_data.y_pos, min=0,
                                                                         max=self.projector_resolution[1]),
                                                 scale_factor=widgets.FloatSlider(
                                                     value=self.calibration_data.scale_factor, min=0.1, max=6.0,
                                                     step=0.01, continuous_update=False),
                                                 z_range=widgets.IntRangeSlider(
                                                     value=[self.calibration_data.z_range[0],
                                                            self.calibration_data.z_range[1]],
                                                     min=500, max=2000, step=1, continuous_update=False),
                                                 box_width=widgets.IntSlider(value=self.calibration_data.box_dim[0],
                                                                             min=0,
                                                                             max=2000, continuous_update=False),
                                                 box_height=widgets.IntSlider(value=self.calibration_data.box_dim[1],
                                                                              min=0,
                                                                              max=2000, continuous_update=False),
                                                 legend_area=widgets.ToggleButton(
                                                     value=self.calibration_data.legend_area,
                                                     description='display a legend',
                                                     disabled=False,
                                                     button_style='',  # 'success', 'info', 'warning', 'danger' or ''
                                                     tooltip='Description',
                                                     icon='check'),
                                                 legend_x_lim=widgets.IntRangeSlider(
                                                     value=[self.calibration_data.legend_x_lim[0],
                                                            self.calibration_data.legend_x_lim[1]],
                                                     min=0, max=self.projector_resolution[0], step=1,
                                                     continuous_update=False),
                                                 legend_y_lim=widgets.IntRangeSlider(
                                                     value=[self.calibration_data.legend_y_lim[0],
                                                            self.calibration_data.legend_y_lim[1]],
                                                     min=0, max=self.projector_resolution[1], step=1,
                                                     continuous_update=False),
                                                 profile_area=widgets.ToggleButton(
                                                     value=self.calibration_data.profile_area,
                                                     description='display a profile area',
                                                     disabled=False,
                                                     button_style='',  # 'success', 'info', 'warning', 'danger' or ''
                                                     tooltip='display a profile area',
                                                     icon='check'),
                                                 profile_x_lim=widgets.IntRangeSlider(
                                                     value=[self.calibration_data.profile_x_lim[0],
                                                            self.calibration_data.profile_x_lim[1]],
                                                     min=0, max=self.projector_resolution[0], step=1,
                                                     continuous_update=False),
                                                 profile_y_lim=widgets.IntRangeSlider(
                                                     value=[self.calibration_data.profile_y_lim[0],
                                                            self.calibration_data.profile_y_lim[1]],
                                                     min=0, max=self.projector_resolution[1], step=1,
                                                     continuous_update=False),
                                                 hot_area=widgets.ToggleButton(
                                                     value=self.calibration_data.hot_area,
                                                     description='display a hot area for qr codes',
                                                     disabled=False,
                                                     button_style='',  # 'success', 'info', 'warning', 'danger' or ''
                                                     tooltip='display a hot area for qr codes',
                                                     icon='check'),
                                                 hot_x_lim=widgets.IntRangeSlider(
                                                     value=[self.calibration_data.hot_x_lim[0],
                                                            self.calibration_data.hot_x_lim[1]],
                                                     min=0, max=self.projector_resolution[0], step=1,
                                                     continuous_update=False),
                                                 hot_y_lim=widgets.IntRangeSlider(
                                                     value=[self.calibration_data.hot_y_lim[0],
                                                            self.calibration_data.hot_y_lim[1]],
                                                     min=0, max=self.projector_resolution[1], step=1,
                                                     continuous_update=False),
                                                 close_click=widgets.ToggleButton(
                                                     value=False,
                                                     description='Close calibration',
                                                     disabled=False,
                                                     button_style='',  # 'success', 'info', 'warning', 'danger' or ''
                                                     tooltip='Close calibration',
                                                     icon='check'
                                                 )

                                                 )
        IPython.display.display(calibration_widget)


class Projector:
    """

    """


    def __init__(self, resolution=None, create_calibration=False, work_directory='./', refresh=100, input_rescale=True):
        """

        Args:
            resolution:
            create_calibration:
            work_directory:
            refresh:
            input_rescale:

        Returns:
            None

        """

        self.html_filename = "projector" + str(self.id) + ".html"
        self.frame_filename = "frame" + str(self.id) + ".png"
        self.input_filename = 'current_frame.png'
        self.legend_filename = 'legend.png'
        self.hot_filename = 'hot.png'
        self.profile_filename = 'profile.png'
        self.work_directory = work_directory
        self.html_file = None
        self.html_text = None
        self.frame_file = None
        self.drawdate = "false"  # Boolean as string for html, only used for testing.
        self.refresh = refresh  # wait time in ms for html file to load image
        self.input_rescale = input_rescale
        if resolution is None:
            print("no resolution specified. please always provide the beamer resolution on initiation!! For now a resolution of 800x600 is used!")
            resolution = (800, 600) #resolution of the beamer that is changeds later is not passed to the calibration!
        self.resolution = resolution
        if create_calibration is True:
            self.calibration = Calibration(associated_projector=self)
            print("created new calibration:", self.calibration)
        else:
            self.calibration = None

    def set_calibration(self, calibration: Calibration):
        """

        Args:
            calibration:

        Returns:
            None

        """
        self.calibration = calibration

    def calibrate(self):
        # TODO This method should be in the calibration class
        self.calibration.create()

    def start_stream(self):
        """

        Returns:

        """
        # def start_stream(self, html_file=self.html_file, frame_file=self.frame_file):
        if self.work_directory is None:
            self.work_directory = os.getcwd()
        self.html_file = open(os.path.join(self.work_directory, self.html_filename), "w")

        self.html_text = """
            <html>
            <head>
                <style>
                    body {{ margin: 0px 0px 0px 0px; padding: 0px; }} 
                </style>
            <script type="text/JavaScript">
            var url = "output.png"; //url to load image from
            var refreshInterval = {0} ; //in ms
            var drawDate = {1}; //draw date string
            var img;

            function init() {{
                var canvas = document.getElementById("canvas");
                var context = canvas.getContext("2d");
                img = new Image();
                img.onload = function() {{
                    canvas.setAttribute("width", img.width)
                    canvas.setAttribute("height", img.height)
                    context.drawImage(this, 0, 0);
                    if(drawDate) {{
                        var now = new Date();
                        var text = now.toLocaleDateString() + " " + now.toLocaleTimeString();
                        var maxWidth = 100;
                        var x = img.width-10-maxWidth;
                        var y = img.height-10;
                        context.strokeStyle = 'black';
                        context.lineWidth = 2;
                        context.strokeText(text, x, y, maxWidth);
                        context.fillStyle = 'white';
                        context.fillText(text, x, y, maxWidth);
                    }}
                }};
                refresh();
            }}
            function refresh()
            {{
                img.src = url + "?t=" + new Date().getTime();
                setTimeout("refresh()",refreshInterval);
            }}

            </script>
            <title>AR Sandbox output</title>
            </head>

            <body onload="JavaScript:init();">
            <canvas id="canvas"/>
            </body>
            </html>

            """
        self.html_text = self.html_text.format(self.refresh, self.drawdate)
        self.html_file.write(self.html_text)
        self.html_file.close()

        webbrowser.open_new('file://' + str(os.path.abspath((os.path.join(self.work_directory, self.html_filename)))))

    def show(self, input=None, legend_filename=None, profile_filename=None,
             hot_filename=None, rescale=None):
        """

        Args:
            input:
            legend_filename:
            profile_filename:
            hot_filename:
            rescale:

        Returns:

        """

        assert self.calibration is not None, 'Calibration is not set yet. See set_calibration.'

        if input is None:
            input = os.path.join(self.work_directory, self.input_filename)
        if legend_filename is None:
            legend_filename = os.path.join(self.work_directory, self.legend_filename)
        if profile_filename is None:
            profile_filename = os.path.join(self.work_directory,self.profile_filename)
        if hot_filename is None:
            hot_filename = os.path.join(self.work_directory,self.hot_filename)
        if rescale is None: #
            rescale = self.input_rescale

        projector_output = Image.new('RGB', self.resolution)
        frame = Image.open(input)

        if rescale is True:
            projector_output.paste(frame.resize((int(frame.width * self.calibration.calibration_data.scale_factor),
                                              int(frame.height * self.calibration.calibration_data.scale_factor))),
                                (
                                self.calibration.calibration_data.x_pos, self.calibration.calibration_data.y_pos))
        else:
            projector_output.paste(frame, (self.calibration.calibration_data.x_pos, self.calibration.calibration_data.y_pos))

        if self.calibration.calibration_data.legend_area is not False:
            legend = Image.open(legend_filename)
            projector_output.paste(legend, (
            self.calibration.calibration_data.legend_x_lim[0], self.calibration.calibration_data.legend_y_lim[0]))
        if self.calibration.calibration_data.profile_area is not False:
            profile = Image.open(profile_filename)
            projector_output.paste(profile, (self.calibration.calibration_data.profile_x_lim[0],
                                          self.calibration.calibration_data.profile_y_lim[0]))
        if self.calibration.calibration_data.hot_area is not False:
            hot = Image.open(hot_filename)
            projector_output.paste(hot, (
            self.calibration.calibration_data.hot_x_lim[0], self.calibration.calibration_data.hot_y_lim[0]))

        projector_output.save(os.path.join(self.work_directory, 'output_temp.png'))
        os.replace(os.path.join(self.work_directory, 'output_temp.png'), os.path.join(self.work_directory, 'output.png')) #workaround to supress artifacts


class CalibrationData:
    """

    """
    def __init__(self,rot_angle=-180, x_lim=(0,640), y_lim=(0,480), x_pos=0, y_pos=0, scale_factor=1.0, z_range=(800,1400), box_width=1000, box_height=600, legend_area=False,
                      legend_x_lim=(0,20), legend_y_lim=(0,50), profile_area=False, profile_x_lim=(10,200), profile_y_lim=(200,250), hot_area=False, hot_x_lim=(400,450),
                      hot_y_lim=(400,450)):
        """

        Args:
            rot_angle:
            x_lim:
            y_lim:
            x_pos:
            y_pos:
            scale_factor:
            z_range:
            box_width:
            box_height:
            legend_area:
            legend_x_lim:
            legend_y_lim:
            profile_area:
            profile_x_lim:
            profile_y_lim:
            hot_area:
            hot_x_lim:
            hot_y_lim:

        Returns:
            None

        """
        self.rot_angle = rot_angle
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.scale_factor = scale_factor
        self.z_range = z_range
        self.box_width = box_width
        self.box_height = box_height
        self.legend_area = legend_area
        self.legend_x_lim = legend_x_lim
        self.legend_y_lim = legend_y_lim
        self.profile_area = profile_area
        self.profile_x_lim = profile_x_lim
        self.profile_y_lim = profile_y_lim
        self.hot_area = hot_area
        self.hot_x_lim = hot_x_lim
        self.hot_y_lim = hot_y_lim
        self.box_dim=(self.box_width, self.box_height)


class Scale:
    """
    class that handles the scaling of whatever the sandbox shows and the real world sandbox
    self.extent: 3d extent of the model in the sandbox in model units.

    """
    def __init__(self, calibration: Calibration = None, xy_isometric=True, extent=None):
        """

        Args:
            calibration:
            xy_isometric:
            extent:
        """
        if isinstance(calibration, Calibration):
            self.calibration = calibration
        else:
            raise TypeError("you must pass a valid calibration instance")
        self.xy_isometric = xy_isometric
        self.scale = [None, None, None]
        self.pixel_size = [None, None]
        self.pixel_scale = [None, None]
        self.output_res = None

        if extent is None:  # extent should be array with shape (6,) or convert to list?
            self.extent = numpy.asarray([
                0.0,
                self.calibration.calibration_data.box_width,
                0.0,
                self.calibration.calibration_data.box_height,
                self.calibration.calibration_data.z_range[0],
                self.calibration.calibration_data.z_range[1],
                ])

        else:
            self.extent = numpy.asarray(extent)  # check: array with 6 entries!

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

        self.output_res = (self.calibration.calibration_data.x_lim[1] -
                           self.calibration.calibration_data.x_lim[0],
                           self.calibration.calibration_data.y_lim[1] -
                           self.calibration.calibration_data.y_lim[0])
        self.pixel_scale[0] = float(self.extent[1] - self.extent[0]) / float(self.output_res[0])
        self.pixel_scale[1] = float(self.extent[3] - self.extent[2]) / float(self.output_res[1])
        self.pixel_size[0] = float(self.calibration.calibration_data.box_width) / float(self.output_res[0])
        self.pixel_size[1] = float(self.calibration.calibration_data.box_height) / float(self.output_res[1])

        # TODO: change the extrent in place!! or create a new extent object that stores the extent after that modification.
        if self.xy_isometric == True:  # model is extended in one horizontal direction to fit  into box while the scale
            # in both directions is maintained
            print("Aspect ratio of the model is fixed in XY")
            if self.pixel_scale[0] >= self.pixel_scale[1]:
                self.pixel_scale[1] = self.pixel_scale[0]
                print("Model size is limited by X dimension")
            else:
                self.pixel_scale[0] = self.pixel_scale[1]
                print("Model size is limited by Y dimension")

        self.scale[0] = self.pixel_scale[0]/self.pixel_size[0]
        self.scale[1] = self.pixel_scale[1]/self.pixel_size[1]
        self.scale[2] = float(self.extent[5] - self.extent[4]) / (
                self.calibration.calibration_data.z_range[1] -
                self.calibration.calibration_data.z_range[0])
        print("scale in Model units/ mm (X,Y,Z): " + str(self.scale))

    # TODO: manually define zscale and either lower or upper limit of Z, adjust rest accordingly.


class Grid:
    """
    class for grid objects. a grid stores the 3D coordinate of each pixel recorded by the kinect in model coordinates
    a calibration object must be provided, it is used to crop the kinect data to the area of interest
    TODO:  The cropping should be done in the kinect class, with calibration_data passed explicitly to the method! Do this for all the cases where calibration data is needed!
    """
    def __init__(self, calibration=None, scale=None,):
        """

        Args:
            calibration:
            scale:

        Returns:
            None

        """
        if isinstance(calibration, Calibration):
            self.calibration = calibration
        else:
            raise TypeError("you must pass a valid calibration instance")
        if isinstance(scale, Scale):
            self.scale = scale
        else:
            self.scale = Scale(calibration=self.calibration)
            print("no scale provided or scale invalid. A default scale instance is used")
        self.empty_depth_grid=None
        self.create_empty_depth_grid()

    def create_empty_depth_grid(self):
        """
        Sets up XY grid (Z is empty that is the name coming from)

        Returns:

        """

        grid_list = []
        self.output_res = (self.calibration.calibration_data.x_lim[1] -
                           self.calibration.calibration_data.x_lim[0],
                           self.calibration.calibration_data.y_lim[1] -
                           self.calibration.calibration_data.y_lim[0])
        """compare:
        for x in range(self.output_res[1]):
            for y in range(self.output_res[0]):
                grid_list.append([y * self.scale.pixel_scale[1] + self.scale.extent[2], x * self.scale.pixel_scale[0] + self.scale.extent[0]])
        """

        for y in range(self.output_res[1]):
            for x in range(self.output_res[0]):
                grid_list.append([x * self.scale.pixel_scale[0] + self.scale.extent[0],
                                  y * self.scale.pixel_scale[1] + self.scale.extent[2]])

        empty_depth_grid = numpy.array(grid_list)
        self.empty_depth_grid = empty_depth_grid
        self.depth_grid = None #I know, this should have thew right type.. anyway.
        print("the shown extent is ["+str(self.empty_depth_grid[0, 0]) + ", " +
                                      str(self.empty_depth_grid[-1, 0]) + ", " +
                                      str(self.empty_depth_grid[0, 1]) + ", " +
                                      str(self.empty_depth_grid[-1, 1]) + "] "
              )

        # return self.empty_depth_grid

    def update_grid(self, depth):
        """
        Appends the z (depth) coordinate to the empty depth grid.
        this has to be done every frame while the xy coordinates only change if the calibration or model extent is changed.
        For performance reasons these steps are therefore separated.

        Args:
            depth:

        Returns:

        """

        # TODO: is this flip still necessary?
        depth = numpy.fliplr(depth)  ##dirty workaround to get the it running with new gempy version.
        filtered_depth = numpy.ma.masked_outside(depth, self.calibration.calibration_data.z_range[0],
                                                 self.calibration.calibration_data.z_range[1])
        scaled_depth = self.scale.extent[5] - (
                (filtered_depth - self.calibration.calibration_data.z_range[0]) / (
                self.calibration.calibration_data.z_range[1] -
                self.calibration.calibration_data.z_range[0]) * (self.scale.extent[5] - self.scale.extent[4]))
        rotated_depth = scipy.ndimage.rotate(scaled_depth, self.calibration.calibration_data.rot_angle,
                                             reshape=False)
        cropped_depth = rotated_depth[self.calibration.calibration_data.y_lim[0]:
                                      self.calibration.calibration_data.y_lim[1],
                        self.calibration.calibration_data.x_lim[0]:
                        self.calibration.calibration_data.x_lim[1]]

        flattened_depth = numpy.reshape(cropped_depth, (numpy.shape(self.empty_depth_grid)[0], 1))
        depth_grid = numpy.concatenate((self.empty_depth_grid, flattened_depth), axis=1)

        self.depth_grid = depth_grid


class Contour: #TODO: change the whole thing to use keyword arguments!!
    """
    class to handle contour lines in the sandbox. contours can shpow depth or anything else.
    TODO: pass on keyword arguments to the plot and label functions for more flexibility

    """
    def __init__(self, start, end, step, show=True, show_labels=False, linewidth=1.0, colors=[(0, 0, 0, 1.0)],
                 inline=0, fontsize=15, label_format='%3.0f'):
        """

        Args:
            start:
            end:
            step:
            show:
            show_labels:
            linewidth:
            colors:
            inline:
            fontsize:
            label_format:

        Returns:
            None

        """
        self.start = start
        self.end = end
        self.step = step
        self.show = show
        self.show_labels = show_labels
        self.linewidth = linewidth
        self.colors = colors
        self.levels = numpy.arange(self.start, self.end, self.step)
        self.contours = None
        self.data = None # Data has to be updated for each frame

        # label attributes:
        self.inline = inline
        self.fontsize = fontsize
        self.label_format = label_format


class Plot:
    """
    handles the plotting of a sandbox model

    """
    def __init__(self,  calibration=None, cmap=None, norm=None, lot=None, outfile=None):
        """

        Args:
            calibration:
            cmap:
            norm:
            lot:
            outfile:

        Returns:
            None

        """
        if isinstance(calibration, Calibration):
            self.calibration = calibration
        else:
            raise TypeError("you must pass a valid calibration instance")

        self.cmap = cmap
        self.norm = norm
        self.lot = lot
        self.output_res = (
            self.calibration.calibration_data.x_lim[1] -
            self.calibration.calibration_data.x_lim[0],
            self.calibration.calibration_data.y_lim[1] -
            self.calibration.calibration_data.y_lim[0]
                           )

        self.h = self.calibration.calibration_data.scale_factor * (self.output_res[1]) / 100.0
        self.w = self.calibration.calibration_data.scale_factor * (self.output_res[0]) / 100.0
        self.fig = None
        self.ax = None

        self.outfile = outfile

    def add_contours(self, contour, data):
        """
        renders contours to the current plot object. \
        The data has to come in a specific shape as needed by the matplotlib contour function.
        we explicity enforce to provide X and Y at this stage (you are welcome to change this)

        Args:
            contour: a contour instance
            data:  a list with the form x,y,z
                x: list of the coordinates in x direction (e.g. range(Scale.output_res[0])
                y: list of the coordinates in y direction (e.g. range(Scale.output_res[1])
                z: 2D array-like with the values to be contoured

        Returns:

        """

        if contour.show is True:
            contour.contours = self.ax.contour(data[0], data[1], data[2], levels=contour.levels,
                                           linewidths=contour.linewidth, colors=contour.colors)
            if contour.show_labels is True:
                self.ax.clabel(contour.contours, inline=contour.inline, fontsize=contour.fontsize, fmt=contour.label_format)

    def create_empty_frame(self):
        """

        Returns:

        """
        self.fig = plt.figure(figsize=(self.w, self.h), dpi=100, frameon=False)
        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        self.ax.set_axis_off()
        self.fig.add_axes(self.ax)

    def render_frame(self, rasterdata):
        """

        Args:
            rasterdata:

        Returns:

        """
        self.create_empty_frame()
        self.block = rasterdata.reshape((self.output_res[1], self.output_res[0]))
        self.ax.pcolormesh(self.block, cmap=self.cmap, norm=self.norm)

    def add_lith_contours(self, block, levels=None):
        """

        Args:
            block:
            levels:

        Returns:

        """
        plt.contourf(block, levels=levels, cmap=self.cmap, norm=self.norm, extend="both")

    def save(self, outfile=None):
        """

        Args:
            outfile:

        Returns:

        """
        if outfile is None:
            if self.outfile is None:
                print("no outfile provided. try the default output file name 'current_frame.png' ")
                plt.show(self.fig)
                plt.close()
                pass
            else:
                outfile = self.outfile

        self.outfile = outfile
        self.fig.savefig (self.outfile, pad_inches=0)
        plt.close(self.fig)

    def create_legend(self):
        """

        Returns:

        """
        pass


class GeoMapModule:
    """

    """
    # TODO: When we move GeoMapModule import gempy just there

    def __init__(self, geo_model, grid: Grid, geol_map: Plot, work_directory=None):
        """

        Args:
            geo_model:
            grid:
            geol_map:
            work_directory:

        Returns:
            None

        """


        self.geo_model = geo_model
        self.kinect_grid = grid
        self.geol_map = geol_map
        self.work_directory = work_directory

        self.fault_line = self.create_fault_line(0, self.geo_model.geo_data_res.n_faults+0.5001)
        self.main_contours = self.create_main_contours(self.kinect_grid.scale.extent[4],
                                                       self.kinect_grid.scale.extent[5])
        self.sub_contours = self.create_sub_contours(self.kinect_grid.scale.extent[4],
                                                     self.kinect_grid.scale.extent[5])

        self.x_grid = range(self.kinect_grid.scale.output_res[0])
        self.y_grid = range(self.kinect_grid.scale.output_res[1])

        self.plot_topography = True
        self.plot_faults = True

    def compute_model(self, kinect_array):
        """

        Args:
            kinect_array:

        Returns:

        """
        self.kinect_grid.update_grid(kinect_array)
        sol = gp.compute_model_at(self.kinect_grid.depth_grid, self.geo_model)
        lith_block = sol[0][0]
        fault_blocks = sol[1][0::2]
        block = lith_block.reshape((self.kinect_grid.scale.output_res[1],
                                    self.kinect_grid.scale.output_res[0]))

        return block, fault_blocks

    # TODO: Miguel: outfile folder should follow by default whatever is set in projection!
    # TODO: Temporal fix. Eventually we need a container class or metaclass with this data
    def render_geo_map(self, block, fault_blocks, outfile=None):
        """

        Args:
            block:
            fault_blocks:
            outfile:

        Returns:

        """
        if outfile is None:
            outfile = os.path.join(self.work_directory, "current_frame.png")

        self.geol_map.render_frame(block)

        elevation = self.kinect_grid.depth_grid.reshape((self.kinect_grid.scale.output_res[1],
                                                         self.kinect_grid.scale.output_res[0], 3))[:, :, 2]
        # This line is for GemPy 1.2: fault_data = sol.fault_blocks.reshape((scalgeol_map.outfilee.output_res[1],
        # scale.output_res[0]))

        if self.plot_faults is True:
            for fault in fault_blocks:
                fault = fault.reshape((self.kinect_grid.scale.output_res[1], self.kinect_grid.scale.output_res[0]))
                self.geol_map.add_contours(self.fault_line, [self.x_grid, self.y_grid, fault])
        if self.plot_topography is True:
            self.geol_map.add_contours(self.main_contours, [self.x_grid, self.y_grid, elevation])
            self.geol_map.add_contours(self.sub_contours, [self.x_grid, self.y_grid, elevation])

        self.geol_map.save(outfile=outfile)

    def create_fault_line(self,
                          start=0.5,
                          end=50.5, #TODO Miguel:increase?
                          step=1.0,
                          linewidth=3.0,
                          colors=[(1.0, 1.0, 1.0, 1.0)]):
        """

        Args:
            start:
            end:
            step:
            linewidth:
            colors:

        Returns:

        """

        self.fault_line = Contour(start=start, end=end, step=step, linewidth=linewidth,
                                  colors=colors)

        return self.fault_line

    def create_main_contours(self, start, end, step=100, linewidth=1.0,
                                  colors=[(0.0, 0.0, 0.0, 1.0)], show_labels=True):
        """

        Args:
            start:
            end:
            step:
            linewidth:
            colors:
            show_labels:

        Returns:

        """

        self.main_contours = Contour(start=start,
                                     end=end,
                                     step=step,
                                     show_labels=show_labels,
                                     linewidth=linewidth, colors=colors)
        return self.main_contours

    def create_sub_contours(self,
                            start,
                            end,
                            step=25,
                            linewidth=0.8,
                            colors=[(0.0, 0.0, 0.0, 0.8)],
                            show_labels=False
                            ):
        """

        Args:
            start:
            end:
            step:
            linewidth:
            colors:
            show_labels:

        Returns:

        """

        self.sub_contours = Contour(start=start, end=end, step=step, linewidth=linewidth, colors=colors, show_labels=show_labels)
        return self.sub_contours

    def export_topographic_map(self, output="topographic_map.pdf"):
        """

        Args:
            output:

        Returns:

        """
        self.geol_map.create_empty_frame()
        elevation = self.kinect_grid.depth_grid.reshape((self.kinect_grid.scale.output_res[1],
                                                         self.kinect_grid.scale.output_res[0], 3))[:, :, 2]
        self.geol_map.add_contours(self.main_contours, [self.x_grid, self.y_grid, elevation])
        self.geol_map.add_contours(self.sub_contours, [self.x_grid, self.y_grid, elevation])
        self.geol_map.save(outfile=output)

    def export_geological_map(self, kinect_array, output="geological_map.pdf"):
        """

        Args:
            kinect_array:
            output:

        Returns:

        """

        print("there is still a bug in the map that causes the uppermost lithology to be displayed in the basement"
              " color. Unfortunately we do not have a quick fix for this currently... Sorry! Please fix the map "
              "yourself, for example using illustrator")

        lith_block, fault_blocks = self.compute_model(kinect_array)

        # This line is for GemPy 1.2: lith_block = sol.lith_block.reshape((scale.output_res[1], scale.output_res[0]))

        self.geol_map.create_empty_frame()

        lith_levels = self.geo_model.potential_at_interfaces[-1].sort()
        self.geol_map.add_lith_contours(lith_block, levels=lith_levels)

        elevation = self.kinect_grid.depth_grid.reshape((self.kinect_grid.scale.output_res[1],
                                                         self.kinect_grid.scale.output_res[0], 3))[:, :, 2]
        # This line is for GemPy 1.2: fault_data = sol.fault_blocks.reshape((scalgeol_map.outfilee.output_res[1],
        # scale.output_res[0]))

        if self.plot_faults is True:
            for fault in fault_blocks:
                fault = fault.reshape((self.kinect_grid.scale.output_res[1], self.kinect_grid.scale.output_res[0]))
                self.geol_map.add_contours(self.fault_line, [self.x_grid, self.y_grid, fault])

        if self.plot_topography is True:
            self.geol_map.add_contours(self.main_contours, [self.x_grid, self.y_grid, elevation])
            self.geol_map.add_contours(self.sub_contours, [self.x_grid, self.y_grid, elevation])

        self.geol_map.save(outfile=output)


class SandboxThread:
    """
    container for modules that handles threading. any kind of module can be loaded, as long as it contains a 'setup' and 'render_frame" method!
    """

    def __init__(self, module, kinect, projector, path=None):
        """

        Args:
            module:
            kinect:
            projector:
            path:
        """
        self.module = module
        self.kinect = kinect
        self.projector = projector
        self.path = path
        self.thread = None
        self.lock = threading.Lock()
        self.stop_thread = False

    def loop(self):
        """

        Returns:

        """
        while self.stop_thread is False:
            depth = self.kinect.get_filtered_frame()
            with self.lock:
                # TODO: Making the next two lines agnostic from GemPy
                lith, fault = self.module.compute_model(depth)
                self.module.render_geo_map(lith, fault, outfile=self.path)
                self.projector.show()

    def run(self):
        """

        Returns:

        """
        self.stop_thread = False
        self.thread = threading.Thread(target=self.loop, daemon=None)
        self.thread.start()
        # with thread and thread lock move these to main sandbox

    def pause(self):
        """

        Returns:

        """
        self.lock.release()

    def resume(self):
        """

        Returns:

        """
        self.lock.acquire()

    def kill(self):
        """

        Returns:

        """
        self.stop_thread = True
        try:
            self.lock.release()
        except:
            pass


class ArucoMarkers:
    """
    class to detect Aruco markers in the kinect data (IR and RGB)
    An Area of interest can be specified, markers outside this area will be ignored
    TODO: run as loop in a thread, probably implement in API
    """
    def __init__(self, aruco_dict=None, Area=None):
        if not aruco_dict:
            self.aruco_dict = #set the default dictionary here
        else:
            self.aruco_dict = aruco_dict
        self.Area = Area # set a square Area of interest here (Hot-Area)
        self.dict_markers_current = {} #  markers that were detected in the last frame
        self.dict_markers_all = {} # all markers ever detected with their last known position and timestamp
        self.lock = threading.Lock #thread lock object to avoid read-write collisions in multithreading.

    def find_markers_ir(self, kinect :KinectV2):
        pass

    def find_markers_rgb(self,kinect :KinectV2):
        pass

    def update_dict_markers_all(self):
        pass






