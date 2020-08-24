import matplotlib.pyplot as plt
import matplotlib
import numpy
import panel as pn
pn.extension()
import weakref


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
        self._cmap = None
        self.norm = norm
        self.lot = lot  # TODO: Future feature
        self.col = None
        self._col = None #weakreference of self.col
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
        ax = sb_params.get('ax')
        ax.texts = [] #TODO: if this is not cleared then the labels breaks the thread
        data = sb_params.get('frame')
        cmap = sb_params.get('cmap')
        norm = sb_params.get('norm')
        extent = sb_params.get('extent')
        self.vmin = extent[-2]
        self.vmax = extent[-1]

        if active and self.active:
            if self._col is not None and self._col() not in ax.images:
                self.col = None
            if self.col is None:# and self.active and active:
            #if len(ax.images)==0:
                self.render_frame(data, ax, vmin=self.vmin, vmax=self.vmax, extent=extent[:4])
            else:
                self.set_data(data)
                self.set_cmap(cmap, 'k', 'k', 'k')
                self.set_norm(norm)
                self.set_extent(extent)
                sb_params['cmap'] = self.cmap
        else:
            #ax.images[0].remove()
            if self.col is not None:
                self.col.remove()
                self.col = None
            if self._col is not None and self._col() in ax.images:
                ax.images.remove(self._col)
        return sb_params

    def set_extent(self, extent):
        self.col.set_extent(extent[:4])

    def set_norm(self, norm):
        #if norm is None:
        #    norm = matplotlib.colors.Normalize(vmin=None, vmax=None, clip=False)
        self.norm = norm
        if self.norm is not None:
            self.col.set_norm(norm)

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
        if isinstance(cmap, str):
            cmap = plt.cm.get_cmap(cmap)

        if self._cmap is not None and self._cmap.name != cmap.name:
            cmap = self._cmap
            self._cmap = None

        if over is not None:
            cmap.set_over(over, 1.0)
        if under is not None:
            cmap.set_under(under, 1.0)
        if bad is not None:
            cmap.set_bad(bad, 1.0)
        self.cmap = cmap
        self.col.set_cmap(cmap)
        return None

    #def set_array(self, data):
    #    """
    #    Change the numpy array that is being plotted without the need to errase the pcolormesh figure
    #    Args:
    #        data:
    #    Returns:
    #    """
    #    self.col.set_array(data.ravel())
    #    return None

    def set_data(self, data):
        """
        Change the numpy array that is being plotted without the need to errase the imshow figure
        Args:
            data:
        Returns:
        """
        #masked = numpy.ma.masked_outside(data, self.vmin, self.vmax )
        self.col.set_data(data)
        self.col.set_clim(vmin=self.vmin, vmax=self.vmax)
        #self.col.set_aspect('auto')
        return None

    def render_frame(self, data, ax, vmin=None, vmax=None, extent=None):
        """Renders a new image or actualizes the current one"""
        if vmin is None:
            vmin = self.vmin
        if vmax is None:
            vmax = self.vmax


        #self.col = ax.pcolormesh(data, vmin=vmin, vmax=vmax,
        #                         cmap=self.cmap, norm=self.norm,
        #                         shading='nearest')
        #self.col = matplotlib.image.AxesImage(ax, cmap=self.cmap, norm=self.norm,
                                                #  origin='lower',zorder=-1)
        #self.set_data(data)
        self.col = ax.imshow(data, vmin=vmin, vmax=vmax,
                             cmap=self.cmap, norm=self.norm,
                             origin='lower left', aspect='auto', zorder=-1, extent=extent)
        self._col = weakref.ref(self.col)

        ax.set_axis_off()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        return None

    def delete_image(self):
        """Method to remove the image from the frame"""
        self.col.remove()
        return None

    def show_widgets(self):
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
        #self.lock.acquire()
        self._cmap = plt.cm.get_cmap(event.new)
        #self.set_cmap(cmap, 'k', 'k','k')
        #self.lock.release()
