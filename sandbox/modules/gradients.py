import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LightSource
import numpy
import panel as pn
from .template import ModuleTemplate
from sandbox import set_logger
logger = set_logger(__name__)


class GradientModule(ModuleTemplate):
    """
    Module to display the gradient of the topography and the topography as a vector field.
    """
    def __init__(self, extent: list = None):
        # call parents' class init, use greyscale colormap as standard and extreme color labeling
        pn.extension()
        if extent is not None:
            self.vmin = extent[4]
            self.vmax = extent[5]

        self.extent = extent
        self.frame = None
        # all possible type of gradient plots
        self.grad_type = ['Original',
                          'Gradient dx',
                          'Gradient dy',
                          'Gradient all',
                          'Laplacian',
                          'Lightsource',
                          'White background'
                          ]

        self.active_stream = False
        self.active_vector = False
        self.vector = None
        self.stream = None
        self.current_grad = self.grad_type[0]

        # lightsource parameter
        self.azdeg = 315
        self.altdeg = 4
        self.ve = 0.25
        self.set_lightsource()

        logger.info("GradientModule loaded successfully")

    def update(self, sb_params: dict):
        frame = sb_params.get('frame')
        extent = sb_params.get('extent')
        ax = sb_params.get('ax')
        cmap = sb_params.get("cmap")

        frame, ax, cmap, extent, contour = self.plot(frame, ax, cmap, extent, self.current_grad)

        sb_params['frame'] = frame
        sb_params['ax'] = ax
        sb_params['cmap'] = cmap
        sb_params['extent'] = extent
        if cmap is None:
            sb_params['active_cmap'] = False
            sb_params['active_shading'] = False
        else:
            sb_params['active_cmap'] = True
            sb_params['active_shading'] = False
        if contour is None:
            sb_params['active_contours'] = False
        else:
            sb_params['active_contours'] = True
        return sb_params

    def delete_quiver_ax(self, ax):
        [quiver.remove() for quiver in reversed(ax.collections) if isinstance(quiver, matplotlib.quiver.Quiver)]
        self.vector = None

    def delete_stream_ax(self, ax):
        [arrow.remove() for arrow in reversed(ax.patches) if isinstance(arrow, matplotlib.patches.FancyArrowPatch)]
        [lines.remove() for lines in reversed(ax.collections) if isinstance(lines, matplotlib.collections.LineCollection)]
        self.stream = None

    def plot(self, frame, ax, cmap, extent, current_grad):
        contour = True
        dx, dy = numpy.gradient(frame)
        # Delete everything to plot it new
        if self.vector is not None:  # quiver plot
            self.delete_quiver_ax(ax)
        if self.stream is not None:  # stream plot
            self.delete_stream_ax(ax)

        if self.active_vector:
            _ = self._quiver(frame, dx, dy, ax)

        if self.active_stream:
            _ = self._stream(frame, dx, dy, ax)
            contour = None

        if current_grad == self.grad_type[0]:
            pass
        elif current_grad == self.grad_type[1]:
            frame, extent, cmap = self._dx(dx, extent)
            frame = numpy.clip(frame, extent[-2], extent[-1])
        elif current_grad == self.grad_type[2]:
            frame, extent, cmap = self._dy(dy, extent)
            frame = numpy.clip(frame, extent[-2], extent[-1])
        elif current_grad == self.grad_type[3]:
            frame, extent, cmap = self._dxdy(dx,  dy, extent)
            frame = numpy.clip(frame, extent[-2], extent[-1])
        elif current_grad == self.grad_type[4]:
            frame, extent, cmap = self._laplacian(dx,  dy, extent)
            frame = numpy.clip(frame, extent[-2], extent[-1])
        elif current_grad == self.grad_type[5]:
            frame, cmap = self._lightsource(frame)
            frame = numpy.clip(frame, extent[-2], extent[-1])
        elif current_grad == self.grad_type[6]:
            cmap = None
        else:
            raise NotImplementedError
        return frame, ax, cmap, extent, contour

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
        extent[-1] = 4
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
        # Note: 180 degrees are subtracted because visualization in Sandbox is upside-down
        ls = LightSource(azdeg=self.azdeg - 180, altdeg=self.altdeg)
        cmap = plt.cm.copper
        rgb = ls.shade(frame, cmap=cmap, vert_exag=self.ve, blend_mode='hsv')
        return rgb, cmap

    def _quiver(self, frame, dx, dy, ax, spacing=10):
        xx, yy = frame.shape
        if self.vector is None:
            self.vector = ax.quiver(numpy.arange(spacing, yy - spacing, spacing),
                                    numpy.arange(spacing, xx - spacing, spacing),
                                    dy[spacing:-spacing:spacing, spacing:-spacing:spacing]*-1,
                                    dx[spacing:-spacing:spacing, spacing:-spacing:spacing]*-1,
                                    zorder=3)
        else:
            self.vector.set_UVC(dy[spacing:-spacing:spacing, spacing:-spacing:spacing]*-1,
                                dx[spacing:-spacing:spacing, spacing:-spacing:spacing]*-1)
        cmap = None
        return frame, cmap

    def _stream(self, frame, dx, dy, ax, spacing=10):
        xx, yy = frame.shape
        self.stream = ax.streamplot(numpy.arange(spacing, yy - spacing, spacing),
                                    numpy.arange(spacing, xx - spacing, spacing),
                                    dy[spacing:-spacing:spacing, spacing:-spacing:spacing]*-1,
                                    dx[spacing:-spacing:spacing, spacing:-spacing:spacing]*-1,
                                    zorder=3,
                                    color='blue')
        cmap = None
        contour = None
        return frame, cmap, contour

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

        self._widget_active_vector = pn.widgets.Checkbox(name='Show vector field', value=self.active_vector)
        self._widget_active_vector.param.watch(self._callback_active_vector, 'value',
                                               onlychanged=False)

        self._widget_active_stream = pn.widgets.Checkbox(name='Show stream plot', value=self.active_stream)
        self._widget_active_stream.param.watch(self._callback_active_stream, 'value',
                                               onlychanged=False)

        column = pn.Column("### Plot gradient model",
                           self._widget_gradient,
                           "### Show direction of gradient",
                           self._widget_active_vector,
                           self._widget_active_stream)
        return column

    def show_widgets(self):
        """
           Create and show the widgets associated to this module
           Returns:
               widget
           """
        panel = pn.Row(self.widget_gradient(), self.widget_lightsource())
        return panel

    def _callback_active_vector(self, event):
        self.active_vector = event.new

    def _callback_active_stream(self, event):
        self.active_stream = event.new

    def _callback_lightsource_altdeg(self, event):
        self.altdeg = event.new

    def _callback_lightsource_azdeg(self, event):
        self.azdeg = event.new

    def _callback_gradient(self, event):
        self.set_gradient(event.new)
