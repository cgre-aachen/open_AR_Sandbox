import numpy
import matplotlib

class ContourLinesModule:
    dpi = 100  # make sure that figures can be displayed pixel-precise
    def __init__(self, contours=True, contours_step=100,
                 contours_width=1.0, contours_color='k', contours_label=False,
                 contours_label_inline=True, contours_label_fontsize=15,
                 contours_label_format='%3.0f', minor_contours=True,
                 contours_step_minor=50, contours_width_minor=0.5,
                 extent=None):
            """
            Module for the display and manipulation of contourlines

            Args:
                contours (bool): Flag that enables or disables contours plotting.
                    (default is True)
                contours_step (int): Size of step between contour lines in model units.
                    (default is 100)
                contours_width (float): Width of contour lines.
                    (default is 1.0)
                contours_color (e.g. str): Color of contour lines.
                    (default is 'k')
                contours_label (bool): Flag that enables labels on contour lines.
                    (default is False)
                contours_label_inline (bool): Partly replace underlying contour line or not.
                    (default is True)
                contours_label_fontsize (float or str): Size in points or relative size of contour label.
                    (default is 15)
                contours_label_format (string or dict): Format string for the contour label.
                    (default is %3.0f)
                minor_contours (bool): Flag that enables or disables minor contours plotting.
                    (default is True)
                contours_step_minor (int): Size of step between minor contour lines in model units.
                    (default is 50)
                contours_width_minor (float): Width of minor contour lines.
                    (default is 0.5)
                extent (list): extents of the sandbox to indicate the physical dimensions of it
            """
            self.contours = None

            self.extent = extent
            self.vmin = self.extent[4]
            self.vmax = self.extent[5]

            # flags
            self.contours = contours
            self.minor_contours = minor_contours

            # contours setup
            self.contours_step = contours_step  # levels will be supplied via property function
            self.contours_width = contours_width
            self.contours_color = contours_color
            self.contours_label = contours_label
            self.contours_label_inline = contours_label_inline
            self.contours_label_fontsize = contours_label_fontsize
            self.contours_label_format = contours_label_format

            self.contours_step_minor = contours_step_minor
            self.contours_width_minor = contours_width_minor

    def plot_contour_lines(self, frame, ax):
        self.add_major_contours(frame, ax)
        self.add_minor_contours(frame, ax)
        self.add_label_contours(frame, ax)

    def add_major_contours(self, data, ax, extent=None):
        """Renders contours to the current plot object.
        Uses the different attributes to style contour lines and contour labels.
        """
        self.contours = ax.contour(data,
                                   levels=self.contours_levels,
                                   linewidths=self.contours_width,
                                   colors=self.contours_color,
                                   #extent=extent
                                   )

    def add_minor_contours(self, data, ax, extent=None):
        minor = ax.contour(data,
                           levels=self.contours_levels_minor,
                           linewidths=self.contours_width_minor,
                           colors=self.contours_color,
                           #extent=extent
                           )

    def add_label_contours(self, data, ax, extent=None):
        label = ax.clabel(self.contours,
                          inline=self.contours_label_inline,
                          fontsize=self.contours_label_fontsize,
                          fmt=self.contours_label_format)

    @property
    def contours_levels(self):
        """Returns the current contour levels, being aware of changes in calibration."""

        return numpy.arange(self.vmin, self.vmax, self.contours_step)

    @property
    def contours_levels_minor(self):
        """Returns the current contour levels, being aware of changes in calibration."""

        return numpy.arange(self.vmin, self.vmax, self.contours_step_minor)




"""
        def widgets_plot(self):
            self._create_plot_widgets()

            widgets_countours = pn.WidgetBox(self._widget_plot_contours,
                                             self._widget_plot_step_contours,
                                             self._widget_plot_minorcontours,
                                             self._widget_plot_step_minorcontours,
                                             self._widget_plot_contours_label,
                                             self._widget_plot_contours_label_fontsize
                                             )

            panel = pn.Column("#### <b> Dashboard for plot Visualization </b>",
                              "<b> Colormap </b>",
                              self._widget_plot_colormap,
                              self._widget_plot_cmap,
                              "<b> Contour lines </b>",
                              widgets_countours)
            return panel

        def _create_plot_widgets(self):
            # Colormap
            self._widget_plot_cmap = pn.widgets.Select(name='Choose a colormap', options=plt.colormaps(),
                                                       value=self.cmap.name)
            self._widget_plot_cmap.param.watch(self._callback_plot_cmap, 'value', onlychanged=False)

            self._widget_plot_colormap = pn.widgets.Checkbox(name='Show colormap', value=self.colormap)
            self._widget_plot_colormap.param.watch(self._callback_plot_colormap, 'value',
                                                   onlychanged=False)

            # Countours
            self._widget_plot_contours = pn.widgets.Checkbox(name='Show contours', value=self.contours)
            self._widget_plot_contours.param.watch(self._callback_plot_contours, 'value',
                                                   onlychanged=False)

            self._widget_plot_minorcontours = pn.widgets.Checkbox(name='Show minor contours', value=self.minor_contours)
            self._widget_plot_minorcontours.param.watch(self._callback_plot_minorcontours, 'value',
                                                        onlychanged=False)
            self._widget_plot_step_contours = pn.widgets.Spinner(name='Choose a contour step', value=self.contours_step)
            # self._widget_plot_step_contours = pn.widgets.TextInput(name='Choose a contour step')
            self._widget_plot_step_contours.param.watch(self._callback_plot_step_contours, 'value', onlychanged=False)
            # self._widget_plot_step_contours.value = str(self.contours_step)

            self._widget_plot_step_minorcontours = pn.widgets.Spinner(name='Choose a minor contour step',
                                                                      value=self.contours_step_minor)
            self._widget_plot_step_minorcontours.param.watch(self._callback_plot_step_minorcontours, 'value',
                                                             onlychanged=False)
            # self._widget_plot_step_minorcontours.value = str(self.contours_step_minor)

            self._widget_plot_contours_label = pn.widgets.Checkbox(name='Show contours label',
                                                                   value=self.contours_label)
            self._widget_plot_contours_label.param.watch(self._callback_plot_contours_label, 'value',
                                                         onlychanged=False)

            self._widget_plot_contours_label_fontsize = pn.widgets.Spinner(name='set a contour label fontsize',
                                                                           value=self.contours_label_fontsize)
            self._widget_plot_contours_label_fontsize.param.watch(self._callback_plot_contours_label_fontsize, 'value',
                                                                  onlychanged=False)
            # self._widget_plot_contours_label_fontsize.value = str(self.contours_label_fontsize)

            # norm #TODO: normalize

        def _callback_plot_colormap(self, event):
            self.colormap = event.new

        def _callback_plot_contours(self, event):
            self.contours = event.new

        def _callback_plot_minorcontours(self, event):
            self.minor_contours = event.new

        def _callback_plot_cmap(self, event):
            self.cmap = plt.cm.get_cmap(event.new)

        def _callback_plot_step_contours(self, event):
            self.contours_step = event.new

        def _callback_plot_step_minorcontours(self, event):
            self.contours_step_minor = event.new

        def _callback_plot_contours_label(self, event):
            self.contours_label = event.new

        def _callback_plot_contours_label_fontsize(self, event):
            self.contours_label_fontsize = event.new

        ##### Widgets for aruco plotting

        def widgets_aruco(self):
            self._create_aruco_widgets()
            widgets = pn.WidgetBox(self._widget_aruco_scatter,
                                   self._widget_aruco_annotate,
                                   self._widget_aruco_connect,
                                   self._widget_aruco_color)

            panel = pn.Column("<b> Dashboard for aruco Visualization </b>", widgets)
            return panel

        def _create_aruco_widgets(self):
            self._widget_aruco_scatter = pn.widgets.Checkbox(name='Show location aruco', value=self.aruco_scatter)
            self._widget_aruco_scatter.param.watch(self._callback_aruco_scatter, 'value',
                                                   onlychanged=False)

            self._widget_aruco_annotate = pn.widgets.Checkbox(name='Show aruco id', value=self.aruco_annotate)
            self._widget_aruco_annotate.param.watch(self._callback_aruco_annotate, 'value',
                                                    onlychanged=False)

            self._widget_aruco_connect = pn.widgets.Checkbox(name='Show line connecting arucos',
                                                             value=self.aruco_connect)
            self._widget_aruco_connect.param.watch(self._callback_aruco_connect, 'value',
                                                   onlychanged=False)

            self._widget_aruco_color = pn.widgets.Select(name='Choose a color', options=[*mcolors.cnames.keys()],
                                                         value=self.aruco_color)
            self._widget_aruco_color.param.watch(self._callback_aruco_color, 'value', onlychanged=False)

        def _callback_aruco_scatter(self, event):
            self.aruco_scatter = event.new

        def _callback_aruco_annotate(self, event):
            self.aruco_annotate = event.new

        def _callback_aruco_connect(self, event):
            self.aruco_connect = event.new

        def _callback_aruco_color(self, event):
            self.aruco_color = event.new
"""

