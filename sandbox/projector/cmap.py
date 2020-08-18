import matplotlib.pyplot as plt
import numpy
import panel as pn
pn.extension()


class CmapModule:
    """
    Class to manage changes in the colormap and plot in the desired projector figure
    """
    def __init__(self, cmap='gist_earth', over=None, under=None,
                 bad=None, norm=None, lot=None, vmin=None, vmax=None,
                 extent=None):
        """
        Initialize the colormap to plot using imshow()
        Args:
            cmap (str or plt.Colormap): Matplotlib colormap, given as name or instance.
            over (e.g. str): Color used for values above the expected data range.
            under (e.g. str): Color used for values below the expected data range.
            bad (e.g. str): Color used for invalid or masked data values.
            norm: Future feature!
            lot: Future feature!
            vmin (float): ...
            vmax (float): ...
            extent (list): ...
        """
        # z-range handling
        self.lock = None  # For locking the multithreading while using bokeh server
        self.extent = extent[:4]
        if vmin is not None:
            self.vmin = vmin
        else:
            self.vmin = extent[4]

        if vmax is not None:
            self.vmax = vmax
        else:
            self.vmax = extent[5]

        self.cmap = plt.cm.get_cmap(cmap)#self.set_cmap(plt.cm.get_cmap(cmap), over, under, bad)

        self.norm = norm  # TODO: Future feature
        self.lot = lot  # TODO: Future feature
        self.col = None
        self.active = True

    def update(self, sb_params: dict):# data, extent, ax, cmap, norm):
        """
        if self.active:
            self.set_data(data)
            self.set_cmap(cmap, 'k', 'k', 'k')
        else:
            self.delete_image() #Todo
        """
        active = sb_params.get('active_cmap')
        if active and self.active:
            data = sb_params.get('frame')
            ax = sb_params.get('ax')
            cmap = sb_params.get('cmap')
            norm = sb_params.get('norm')
            extent = sb_params.get('extent')
            self.vmin = extent[-2]
            self.vmax = extent[-1]
            self.set_cmap(cmap, 'k', 'k', 'k')
            self.set_norm(norm)
            self.render_frame(data, ax)

            sb_params['cmap'] = self.cmap

        return sb_params

    def set_norm(self, norm):
        self.norm = norm

    def set_cmap(self, cmap, over=None, under=None, bad=None):
        """
        Methods to mask the values outside the extent
        Args:
            cmap: (matplotlib colormap): colormap to use
            over (e.g. str): Color used for values above the expected data range.
            under (e.g. str): Color used for values below the expected data range.
            bad (e.g. str): Color used for invalid or masked data values.

        Returns:
        """
        if over is not None:
            cmap.set_over(over, 1.0)
        if under is not None:
            cmap.set_under(under, 1.0)
        if bad is not None:
            cmap.set_bad(bad, 1.0)
        self.cmap = cmap
        self.col.set_cmap(cmap)
        return None

    def set_data(self, data):
        """
        Change the numpy array that is being plotted without the need to errase the arrays
        Args:
            data:
        Returns:
        """
        self.col.set_array(data.ravel())
        return None

    def render_frame(self, data, ax, vmin=None, vmax=None, extent=None):
        """Renders a new image or actualizes the current one"""
        if vmin is None:
            vmin = self.vmin
        if vmax is None:
            vmax = self.vmax

        self.col = ax.pcolormesh(data, vmin=vmin, vmax=vmax,
                                 cmap=self.cmap, norm=self.norm,
                                 shading='nearest')
        #self.col = ax.imshow(data, vmin=vmin, vmax=vmax,
        #                     cmap=self.cmap, norm=self.norm,
        #                     origin='lower', aspect='auto', zorder=1)
        ax.set_axis_off()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        return None

    def delete_image(self):
        """Method to remove the image from the frame"""
        self.col.remove()
        return None

    def widgets_plot(self):
        self._create_widgets()

        panel = pn.Column("<b> Colormap </b>",
                          self._widget_plot_cmap,
                          self._widget_plot_colormap,
                          )

        return panel

    def _create_widgets(self):
        self._widget_plot_cmap = pn.widgets.Select(name='Choose a colormap', options=plt.colormaps(),
                                                   value=self.cmap.name)
        self._widget_plot_cmap.param.watch(self._callback_plot_cmap, 'value', onlychanged=False)

        self._widget_plot_colormap = pn.widgets.Checkbox(name='Show colormap', value=self.active)
        self._widget_plot_colormap.param.watch(self._callback_plot_colormap, 'value',
                                              onlychanged=False)

        return True

    def _callback_plot_colormap(self, event):
        self.active = event.new

    def _callback_plot_cmap(self, event):
        cmap = plt.cm.get_cmap(event.new)
        self.set_cmap(cmap, 'k', 'k','k')
