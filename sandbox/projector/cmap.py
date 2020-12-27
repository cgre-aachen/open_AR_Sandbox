import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import copy
import panel as pn
import weakref
pn.extension()


class CmapModule:
    """
    Class to manage changes in the colormap and plot in the desired projector figure
    """
    def __init__(self, cmap='gist_earth', norm=None, vmin=None, vmax=None, extent=None):
        """
        Initialize the colormap to plot using imshow()
        Args:
            cmap (str or plt.Colormap): Matplotlib colormap, given as name or instance.
            norm: Apply norm to imshow
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

        self.cmap = plt.cm.get_cmap(cmap)
        self._cmap = None
        self.norm = norm
        self.col = None
        self._col = None  # weakreference of self.col
        self.active = True

        # Relief shading
        self.relief_shading = True

        self.azdeg = 315
        self.altdeg = 45
        self.ve = 0.25

    def update(self, sb_params: dict):
        active = sb_params.get('active_cmap')
        active_shade = sb_params.get('active_shading')
        ax = sb_params.get('ax')
        data = sb_params.get('frame')
        cmap = sb_params.get('cmap')
        norm = sb_params.get('norm')
        extent = sb_params.get('extent')
        self.vmin = extent[-2]
        self.vmax = extent[-1]

        if active_shade and self.relief_shading:
            # Note: 180 degrees are subtracted because visualization in Sandbox is upside-down
            ls = mcolors.LightSource(azdeg=self.azdeg - 180, altdeg=self.altdeg)
            data = ls.shade(data, cmap=self.cmap, vert_exag=self.ve, blend_mode='overlay')

        if active and self.active:
            if self._col is not None and self._col() not in ax.images:
                self.col = None
            if self.col is None:
                self.render_frame(data, ax, vmin=self.vmin, vmax=self.vmax, extent=extent[:4])
            else:
                self.set_data(data)
                self.set_cmap(cmap, 'k', 'k', 'k')
                self.set_norm(norm)
                self.set_extent(extent)
                sb_params['cmap'] = self.cmap
        elif active_shade and self.relief_shading:
            cmap = plt.cm.gray
            if self._col is not None and self._col() not in ax.images:
                self.col = None
            if self.col is None:
                self.render_frame(data, ax, vmin=self.vmin, vmax=self.vmax, extent=extent[:4])
            else:
                self.set_data(data)
                self.set_cmap(cmap, 'k', 'k', 'k')
                self.set_norm(norm)
                self.set_extent(extent)
        else:
            if self.col is not None:
                self.col.remove()
                self.col = None
            if self._col is not None and self._col() in ax.images:
                ax.images.remove(self._col)

        return sb_params

    def set_extent(self, extent):
        self.col.set_extent(extent[:4])

    def set_norm(self, norm):
        # if norm is None:
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
        cmap = copy.copy(cmap)
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
        Change the numpy array that is being plotted without the need to errase the imshow figure
        Args:
            data:
        Returns:
        """
        self.col.set_data(data)
        self.col.set_clim(vmin=self.vmin, vmax=self.vmax)
        return None

    def render_frame(self, data, ax, vmin=None, vmax=None, extent=None):
        """Renders a new image or actualizes the current one"""
        if vmin is None:
            vmin = self.vmin
        if vmax is None:
            vmax = self.vmax

        self.col = ax.imshow(data, vmin=vmin, vmax=vmax,
                             cmap=self.cmap, norm=self.norm,
                             origin='lower', aspect='auto', zorder=-5, extent=extent)
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
                          self._widget_plot_colormap,
                          self._widget_plot_cmap,
                          self._widget_lightsource())
        return panel

    def _create_widgets(self):
        self._widget_plot_cmap = pn.widgets.Select(name='Choose a colormap',
                                                   # use the following line to enable all colormaps
                                                   # options=plt.colormaps(),
                                                   # limit to only specified color maps
                                                   options=['gist_earth', 'terrain', 'ocean', 'seismic',
                                                            'RdBu', "RdBu_r", "Greys", "Greys_r",
                                                            'viridis', 'viridis_r', 'magma', 'magma_r',
                                                            ],
                                                   value=self.cmap.name)
        self._widget_plot_cmap.param.watch(self._callback_plot_cmap, 'value', onlychanged=False)

        self._widget_plot_colormap = pn.widgets.Checkbox(name='Show colormap', value=self.active)
        self._widget_plot_colormap.param.watch(self._callback_plot_colormap, 'value',
                                               onlychanged=False)

        return True

    def _widget_lightsource(self):
        self._widget_relief_shading = pn.widgets.Checkbox(name='Show relief shading',
                                                          value=self.relief_shading)
        self._widget_relief_shading.param.watch(self._callback_relief_shading, 'value',
                                                onlychanged=False)
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

        self._widget_ve = pn.widgets.Spinner(name="Vertical Exageration", value=self.ve,
                                             step=0.01)
        self._widget_ve.param.watch(self._callback_ve, 'value', onlychanged=False)

        widgets = pn.WidgetBox(self._widget_relief_shading,
                               self._widget_azdeg,
                               self._widget_altdeg,
                               self._widget_ve
                               )

        panel = pn.Column("<b> Lightsource </b> ", widgets)
        return panel

    def _callback_plot_colormap(self, event): self.active = event.new

    def _callback_plot_cmap(self, event): self._cmap = plt.cm.get_cmap(event.new)

    def _callback_relief_shading(self, event): self.relief_shading = event.new

    def _callback_ve(self, event): self.ve = event.new

    def _callback_lightsource_altdeg(self, event): self.altdeg = event.new

    def _callback_lightsource_azdeg(self, event): self.azdeg = event.new
