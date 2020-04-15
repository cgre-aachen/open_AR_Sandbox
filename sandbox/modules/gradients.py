import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import numpy
import panel as pn
from .module_main_thread import Module
from sandbox.markers.aruco import ArucoMarkers


class GradientModule(Module):
    """
    Module to display the gradient of the topography and the topography as a vector field.
    """

    def __init__(self, *args, **kwargs):
        # call parents' class init, use greyscale colormap as standard and extreme color labeling
        super().__init__(*args, contours=True, cmap='gist_earth_r', over='k', under='k', **kwargs)
        self.frame = None
        self.grad_type = 1

        #lightsource parameter
        self.azdeg = 315
        self.altdeg = 4
        self.ve= 0.25
        self.set_lightsource()

        self.panel_frame = pn.pane.Matplotlib(plt.figure(), tight=False, height=335)
        plt.close()
        self._create_widget_gradients()

    def setup(self):
        self.frame = self.sensor.get_filtered_frame()
        if self.crop:
            self.frame = self.crop_frame(self.frame)
        self.plot.render_frame(self.frame)
        self.projector.frame.object = self.plot.figure

    def set_gradient(self, i):
        self.grad_type = i

    def set_lightsource(self, azdeg=315, altdeg=4, ve=0.25):
        self.azdeg = azdeg
        self.altdeg = altdeg
        self.ve = ve

    def update(self):
        # with self.lock:
        self.frame = self.sensor.get_filtered_frame()
        if self.crop:
            self.frame = self.crop_frame(self.frame) #crops the extent of the kinect image to the sandbox dimensions
            self.frame = self.clip_frame(self.frame) #clips the z values abopve and below the set vertical extent

        self.plot.ax.cla()  # clear axes to draw new ones on figure
        self.plot_grad()

        # if aruco Module is specified:search, update, plot aruco markers
        if isinstance(self.Aruco, ArucoMarkers):
            self.Aruco.search_aruco()
            self.Aruco.update_marker_dict()
            self.Aruco.transform_to_box_coordinates()
            self.plot.plot_aruco(self.Aruco.aruco_markers)

        self.projector.trigger() # trigger update in the projector class

    def plot_grad(self):
        """Create gradient plot and visualize in sandbox"""
        height_map = self.frame
        # self.frame = numpy.clip(height_map, self.frame.min(), 1500.)
        self.frame = numpy.clip(height_map, self.calib.s_min, self.calib.s_max)
        dx, dy = numpy.gradient(self.frame)
        # calculate curvature
        dxdx, dxdy = numpy.gradient(dx)
        dydx, dydy = numpy.gradient(dy)
        laplacian = dxdx + dydy
        #  hillshade
        ls = LightSource(azdeg=self.azdeg, altdeg=self.altdeg)
        rgb = ls.shade(self.frame, cmap=plt.cm.copper, vert_exag=self.ve, blend_mode='hsv')
        #  for quiver
        xx, yy = self.frame.shape

        if self.grad_type == 1:
            self.plot.ax.pcolormesh(dx, cmap='viridis', vmin=-2, vmax=2)
        if self.grad_type == 2:
            self.plot.ax.pcolormesh(dy, cmap='viridis', vmin=-2, vmax=2)
        if self.grad_type == 3:
            self.plot.ax.pcolormesh(numpy.sqrt(dx**2 + dy**2), cmap='viridis', vmin=0, vmax=5)
        if self.grad_type == 4:
            self.plot.ax.pcolormesh(laplacian, cmap='RdBu_r', vmin=-1, vmax=1)
        if self.grad_type == 5:
            self.plot.ax.imshow(rgb, origin='lower left', aspect='auto') # TODO: use pcolormesh insteead of imshow, this method generates axis to the plot
            self.plot.ax.axis('off')
            self.plot.ax.get_xaxis().set_visible(False)
            self.plot.ax.get_yaxis().set_visible(False)
        if self.grad_type == 6:
            self.plot.ax.quiver(numpy.arange(10, yy-10, 10), numpy.arange(10, xx-10, 10),
                                dy[10:-10:10,10:-10:10], dx[10:-10:10,10:-10:10])
        if self.grad_type == 7:
            self.plot.ax.pcolormesh(laplacian, cmap='RdBu_r', vmin=-1, vmax=1)
            self.plot.ax.quiver(numpy.arange(10, yy-10, 10), numpy.arange(10, xx-10, 10),
                                dy[10:-10:10,10:-10:10], dx[10:-10:10,10:-10:10])

        if self.grad_type == 8:
            self.plot.ax.pcolormesh(laplacian, cmap='RdBu_r', vmin=-1, vmax=1)
            self.plot.ax.streamplot(numpy.arange(10, yy-10, 10), numpy.arange(10, xx-10, 10),
                                dy[10:-10:10,10:-10:10], dx[10:-10:10,10:-10:10])

        # streamplot(X, Y, U, V, density=[0.5, 1])

    # Layouts
    def widget_lightsource(self):
        self._widget_azdeg = pn.widgets.FloatSlider(name='Azimuth',

                                                    value=self.azdeg,
                                                    start=0.0,
                                                    end=360.0)
        self._widget_azdeg.param.watch(self._callback_lightsource_azdeg, 'value')

        self._widget_altdeg = pn.widgets.FloatSlider(name='Altitude',
                                                    value=self.altdeg,
                                                    start=0.0,
                                                    end=90.0)
        self._widget_altdeg.param.watch(self._callback_lightsource_altdeg, 'value')

        widgets=pn.WidgetBox('<b>Azimuth</b>',
                             self._widget_azdeg,
                             '<b>Altitude</b>',
                             self._widget_altdeg)

        panel = pn.Column("### Lightsource ", widgets)
        return panel


    def _callback_lightsource_azdeg(self, event):
      #  self.pause()
        self.azdeg = event.new
      #  self.resume()

    def _callback_lightsource_altdeg(self, event):
      #  self.pause()
        self.altdeg = event.new
      #  self.resume()

    def widget_gradients(self):
        widgets = pn.WidgetBox(self._widget_gradient_dx,
                               self._widget_gradient_dy,
                               self._widget_gradient_sqrt,
                               self._widget_laplacian,
                               self._widget_lightsource,
                               self._widget_vector_field,
                               self._widget_laplacian_vector,
                               self._widget_laplacian_stream)

        panel = pn.Column("### Plot gradient model", widgets)
        return panel

    def _create_widget_gradients(self):

        self._widget_gradient_dx = pn.widgets.Button(name = 'Gradient dx', button_type="success")
        self._widget_gradient_dx.param.watch(self._callback_gradient_dx, 'clicks', onlychanged=False)

        self._widget_gradient_dy = pn.widgets.Button(name = 'Gradient dy', button_type="success")
        self._widget_gradient_dy.param.watch(self._callback_gradient_dy, 'clicks', onlychanged=False)

        self._widget_gradient_sqrt = pn.widgets.Button(name = 'Gradient all',button_type="success")
        self._widget_gradient_sqrt.param.watch(self._callback_gradient_sqrt, 'clicks', onlychanged=False)

        self._widget_laplacian = pn.widgets.Button(name = 'Laplacian', button_type="success")
        self._widget_laplacian.param.watch(self._callback_laplacian, 'clicks', onlychanged=False)

        self._widget_lightsource = pn.widgets.Button(name = 'Lightsource', button_type="success")
        self._widget_lightsource.param.watch(self._callback_lightsource, 'clicks', onlychanged=False)

        self._widget_vector_field = pn.widgets.Button(name = 'Vector field', button_type="success")
        self._widget_vector_field.param.watch(self._callback_vector_field, 'clicks', onlychanged=False)

        self._widget_laplacian_vector = pn.widgets.Button(name = 'Laplacian + Vector field', button_type="success")
        self._widget_laplacian_vector.param.watch(self._callback_laplacian_vector, 'clicks', onlychanged=False)

        self._widget_laplacian_stream = pn.widgets.Button(name = 'Laplacian + Stream',button_type="success")
        self._widget_laplacian_stream.param.watch(self._callback_laplacian_stream, 'clicks', onlychanged=False)

        return True

    def _callback_gradient_dx (self, event):
        self.pause()
        self.set_gradient(1)
        self.resume()

    def _callback_gradient_dy (self, event):
        self.pause()
        self.set_gradient(2)
        self.resume()

    def _callback_gradient_sqrt (self, event):
        self.pause()
        self.set_gradient(3)
        self.resume()

    def _callback_laplacian (self, event):
        self.pause()
        self.set_gradient(4)
        self.resume()

    def _callback_lightsource (self, event):
        self.pause()
        self.set_gradient(5)
        self.resume()

    def _callback_vector_field (self, event):
        self.pause()
        self.set_gradient(6)
        self.resume()

    def _callback_laplacian_vector (self, event):
        self.pause()
        self.set_gradient(7)
        self.resume()

    def _callback_laplacian_stream (self, event):
        self.pause()
        self.set_gradient(8)
        self.resume()
