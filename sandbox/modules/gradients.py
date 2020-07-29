import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import numpy
import panel as pn
pn.extension()

from .template import ModuleTemplate


class GradientModule(ModuleTemplate):
    """
    Module to display the gradient of the topography and the topography as a vector field.
    """

    def __init__(self, *args, extent: list = None, **kwargs):
        # call parents' class init, use greyscale colormap as standard and extreme color labeling
        if extent is not None:
            self.vmin = extent[4]
            self.vmax = extent[5]

        self.extent = extent
        self.frame = None
        #all possible type of gradient plots
        self.grad_type = ['Gradient dx',
                          'Gradient dy',
                          'Gradient all',
                          'Laplacian',
                          'Lightsource',
                          'Vector Field',
                          'Streams',
                          'Laplacian + Vectorfield',
                          'Laplacian + Stream'
                          ]
        self.current_grad = self.grad_type[0]

        #lightsource parameter
        self.azdeg = 315
        self.altdeg = 4
        self.ve = 0.25
        self.set_lightsource()

    def update(self, frame, ax, extent, **kwargs):
        frame = numpy.clip(frame, self.vmin, self.vmax)

        frame, ax, cmap, extent = self.plot(frame, ax, extent, self.current_grad)

        norm = None
        return frame, ax, extent, cmap, norm

    def plot(self, frame, ax, extent, current_grad):
        dx, dy = numpy.gradient(frame)
        if current_grad == self.grad_type[0]:
            frame, extent, cmap = self._dx(dx, extent)
        elif current_grad == self.grad_type[1]:
            frame, extent, cmap = self._dy(dy, extent)
        elif current_grad == self.grad_type[2]:
            frame, extent, cmap = self._dxdy(dx,  dy, extent)
        elif current_grad == self.grad_type[3]:
            frame, extent, cmap = self._laplacian(dx,  dy, extent)
        elif current_grad == self.grad_type[4]:
            frame, cmap = self._lightsource(frame)
        elif current_grad == self.grad_type[5]:
            frame, cmap = self._quiver(frame, dx, dy, ax)
        elif current_grad == self.grad_type[6]:
            frame, cmap = self._stream(frame, dx, dy, ax)
        elif current_grad == self.grad_type[7]:
            frame, extent, cmap = self._laplacian(dx, dy, extent)
            _, _ = self._quiver(frame, dx, dy, ax)
        elif current_grad == self.grad_type[8]:
            frame, extent, cmap = self._laplacian(dx, dy, extent)
            _, _ = self._stream(frame, dx, dy, ax)
        else:
            raise NotImplementedError
        return frame, ax, cmap, extent

    def _dx(self, dx, extent):
        extent[-2] = -2
        extent[-1] = 2
        cmap = plt.get_cmap('viridis')
        return dx, extent, cmap

    def _dy(self, dy, extent):
        extent[-2] = -2
        extent[-1] = 2
        cmap = plt.get_cmap('viridis')
        return dy, extent, cmap

    def _dxdy(self, dx, dy, extent):
        dxdy = numpy.sqrt(dx**2 + dy**2)
        extent[-2] = 0
        extent[-1] = 5
        cmap = plt.get_cmap('viridis')
        return dxdy, extent, cmap

    def _laplacian(self, dx, dy, extent):
        dxdx, dxdy = numpy.gradient(dx)
        dydx, dydy = numpy.gradient(dy)
        laplacian = dxdx + dydy
        extent[-2] = -1
        extent[-1] = 1
        cmap = plt.get_cmap('RdBu_r')
        return laplacian, extent, cmap

    def _lightsource(self, frame):
        ls = LightSource(azdeg=self.azdeg, altdeg=self.altdeg)
        cmap = plt.cm.copper
        rgb = ls.shade(frame, cmap=cmap, vert_exag=self.ve, blend_mode='hsv')
        return rgb, cmap

    def _quiver(self, frame, dx, dy, ax):
        xx, yy = frame.shape
        ax.quiver(numpy.arange(10, yy - 10, 10), numpy.arange(10, xx - 10, 10),
                           dy[10:-10:10, 10:-10:10], dx[10:-10:10, 10:-10:10])
        cmap = None
        #frame = None
        return frame, cmap

    def _stream(self, frame, dx, dy, ax):
        xx, yy = frame.shape
        ax.streamplot(numpy.arange(10, yy - 10, 10), numpy.arange(10, xx - 10, 10),
                                dy[10:-10:10, 10:-10:10], dx[10:-10:10, 10:-10:10])
        cmap = None
        #frame = None
        return frame, cmap

    def set_gradient(self, i):
        """
        Change implicit between all gradient types.
        Args:
            i: string indicating the number to run.
        Returns:

        """
        self.current_grad = i

    def set_lightsource(self, azdeg=315, altdeg=4, ve=0.25):
        """
        modify the azimuth, altitude and vertical exageration for the lightsource mode
        Args:
            azdeg: 0 - North, 90- East, 180 - south, 270 - West, 360 - North
            altdeg: degree of the sun . 0 - 180 - being 90 the vertical position of the sun
            ve: float, vertical exageration

        Returns:
        """
        self.azdeg = azdeg
        self.altdeg = altdeg
        self.ve = ve

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

        widgets = pn.WidgetBox('<b>Azimuth</b>',
                             self._widget_azdeg,
                             '<b>Altitude</b>',
                             self._widget_altdeg)

        panel = pn.Column("### Lightsource ", widgets)
        return panel


    def widget_gradient(self):
        self._widget_gradient = pn.widgets.RadioBoxGroup(name='Plotting options',
                                                         options=self.grad_type,
                                                         inline=False)
        self._widget_gradient.param.watch(self._callback_gradient, 'value', onlychanged=False)
        column = pn.Column("### Plot gradient model", self._widget_gradient)
        return column

    def show_widgets(self):
        """
           Create and show the widgets associated to this module
           Returns:
               widget
           """
        panel = pn.Row(self.widget_gradient(), self.widget_lightsource())
        return panel

    def _callback_lightsource_altdeg(self, event):
        self.altdeg = event.new

    def _callback_lightsource_azdeg(self, event):
        self.azdeg = event.new

    def _callback_gradient(self, event):
        self.set_gradient(event.new)

