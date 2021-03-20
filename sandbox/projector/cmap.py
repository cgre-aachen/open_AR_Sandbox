import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import copy
import panel as pn
import weakref
from sandbox.projector.shading import LightSource
from sandbox import set_logger
logger = set_logger(__name__)
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
        self.light_source = LightSource()
        self._light_simulation = False

        logger.info("CmapModule loaded successfully")

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
        set_cbar = sb_params.get("set_colorbar")
        set_cbar(self.vmin, self.vmax, cmap, norm)

        if active_shade and self.relief_shading:
            if len(data.shape) > 2:  # 3 Then is an image already
                active_shade = False
            else:
                # Note: (Not really) 180 degrees are subtracted because visualization in Sandbox is upside-down
                ls = mcolors.LightSource(azdeg=self.light_source.azimuth, altdeg=self.light_source.altitude)
                data = ls.shade(data, cmap=self.cmap, vert_exag=self.light_source.ve, blend_mode='overlay')

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
                             origin='lower', aspect='auto', zorder=-500, extent=extent)
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
        panel = pn.Column("###<b>Colormap </b>",
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
                                                    value=self.light_source.azimuth,
                                                    start=0.0,
                                                    end=360.0)
        self._widget_azdeg.param.watch(self._callback_lightsource_azdeg, 'value')

        self._widget_altdeg = pn.widgets.FloatSlider(name='Altitude',
                                                     value=self.light_source.altitude,
                                                     start=0.0,
                                                     end=90.0)
        self._widget_altdeg.param.watch(self._callback_lightsource_altdeg, 'value')

        self._widget_ve = pn.widgets.Spinner(name="Vertical Exageration", value=self.light_source.ve,
                                             step=0.01)
        self._widget_ve.param.watch(self._callback_ve, 'value', onlychanged=False)
        self._widget_manual = pn.widgets.Checkbox(name='Manual configuration',
                                                 value=self.light_source.manual)
        self._widget_manual.param.watch(self._callback_manual, 'value',
                                                onlychanged=False)

        self._widget_address = pn.widgets.TextInput(name='Enter address (e.g. City, Country)',
                                                    value=self.light_source.address)
        self._widget_address.param.watch(self._callback_address, 'value', onlychanged=False)

        self._widget_markdown_city = pn.pane.Markdown(self.light_source.full_address, sizing_mode='scale_width')
        self._widget_markdown_date = pn.pane.Markdown(self.light_source.date.ctime(), sizing_mode='scale_width')
        self._widget_sun = pn.pane.Markdown("<p>Azimuth: %.4f </p>"\
                                            "<p>Altitude: %.4f </p>" % (self.light_source.azimuth,
                                                                        self.light_source.altitude),
                                            sizing_mode='scale_width')
        self._widget_markdown_lat_long = pn.pane.Markdown("<p>Latitude: %.4f </p>"\
                                                          "<p>Longitude: %.4f </p>" % (self.light_source.latitude_deg,
                                                                                       self.light_source.longitude_deg),
                                                          sizing_mode='scale_width')

        # self._widget_date_pick = pn.widgets.DatetimeInput(name='Select date', value=self.light_source.date)
        self._widget_date_pick = pn.widgets.DatePicker(name='Select date (UTC +0)', value=self.light_source.date.date())
        self._widget_date_pick.param.watch(self._callback_date_pick, 'value', onlychanged=False)

        self._widget_hour_pick = pn.widgets.IntSlider(name="Hour", value=self.light_source.date.hour,
                                                      start=0, end=23, width_policy='min')
        self._widget_hour_pick.param.watch(self._callback_hour_pick, 'value', onlychanged=False)

        self._widget_minute_pick = pn.widgets.IntSlider(name="Minute", value=self.light_source.date.minute,
                                                        start=0, end=59, width_policy='min')
        self._widget_minute_pick.param.watch(self._callback_minute_pick, 'value', onlychanged=False)

        self._widget_second_pick = pn.widgets.IntSlider(name="Second", value=self.light_source.date.second,
                                                        start=0, end=59, width_policy='min')
        self._widget_second_pick.param.watch(self._callback_second_pick, 'value', onlychanged=False)

        self._widget_days_simulation = pn.widgets.Checkbox(name='Start day simulation increasing by hour',
                                                 value=self.light_source.simulation)
        self._widget_days_simulation.param.watch(self._callback_day_simulation, 'value',
                                                onlychanged=False)

        widgets = pn.WidgetBox(self._widget_manual,
                               self._widget_azdeg,
                               self._widget_altdeg,
                               self._widget_ve
                               )
        widgets2 = pn.WidgetBox(self._widget_address,
                                self._widget_date_pick,
                                pn.Row(self._widget_hour_pick,
                                       self._widget_minute_pick,
                                       self._widget_second_pick,
                                       width_policy='min'),
                                self._widget_markdown_date,
                                self._widget_sun,
                                self._widget_markdown_lat_long,
                                self._widget_markdown_city)

        widget3 = pn.WidgetBox(self._widget_days_simulation,
                               # self._widget_markdown_date,
                               # self._widget_sun,
                               # self._widget_markdown_lat_long,
                               )

        tab = pn.Tabs(("Manual", widgets),
                      ("Geo-location", widgets2),
                      ("Day simulation", widget3))

        panel = pn.Column("<b> Lightsource </b> ", self._widget_relief_shading, tab)
        return panel

    def _trigger_info(self):
        self.light_source.manual = False
        self._widget_manual.value = False
        self._widget_sun.object = "<p>Azimuth: %.4f </p>" \
                                  "<p>Altitude: %.4f </p>" % (self.light_source.azimuth,
                                                              self.light_source.altitude)
        self._widget_markdown_lat_long.object = "<p>Latitude: %.4f</p>" \
                                                "<p>Longitude: %.4f</p>" % (self.light_source.latitude_deg,
                                                                            self.light_source.longitude_deg)
        self._widget_markdown_date.object = self.light_source.date.ctime()
        self._widget_markdown_city.object = self.light_source.full_address

    def _callback_day_simulation(self, event):
        self.light_source.simulation = event.new
        self._light_simulation = event.new

    def _callback_hour_pick(self, event):
        self.light_source.date = self.light_source.date.replace(hour=event.new)
        self._trigger_info()

    def _callback_minute_pick(self, event):
        self.light_source.date = self.light_source.date.replace(minute=event.new)
        self._trigger_info()

    def _callback_second_pick(self, event):
        self.light_source.date = self.light_source.date.replace(second=event.new)
        self._trigger_info()

    def _callback_date_pick(self, event):
        self.light_source.set_datetime(date=event.new)
        self._trigger_info()

    def _callback_address(self, event):
        self.light_source.set_address(event.new)
        self.light_source.set_latitude_longitude()
        self._trigger_info()


    def _callback_manual(self, event): self.light_source.manual = event.new

    def _callback_plot_colormap(self, event): self.active = event.new

    def _callback_plot_cmap(self, event): self._cmap = plt.cm.get_cmap(event.new)

    def _callback_relief_shading(self, event): self.relief_shading = event.new

    def _callback_ve(self, event): self.light_source.set_ve(event.new)

    def _callback_lightsource_altdeg(self, event): self.light_source.set_altitude(event.new)

    def _callback_lightsource_azdeg(self, event): self.light_source.set_azimuth(event.new)
