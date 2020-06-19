import matplotlib.pyplot as plt
import numpy
import matplotlib.colors as mcolors
import panel as pn
from matplotlib.figure import Figure


class Plot:  # TODO: include function to take screenshot

    dpi = 100  # make sure that figures can be displayed pixel-precise

    def __init__(self, calibrationdata, contours=True, margins=False,
                 vmin=None, vmax=None, cmap=None, over=None, under=None,
                 bad=None, norm=None, lot=None, margin_color='r', margin_alpha=0.5,
                 contours_step=100, contours_width=1.0, contours_color='k',
                 contours_label=False, contours_label_inline=True,
                 contours_label_fontsize=15, contours_label_format='%3.0f', #old args
                 model=None, show_faults=True, show_lith=True, #new args
                 #marker_position=None,
                 minor_contours=False, contours_step_minor=50,
                 contours_width_minor = 0.5,
                 aruco_connect=True, aruco_scatter = True, aruco_annotate = True, aruco_color = 'red'):
        """Creates a new plot instance.

        Regularly, this creates at least a raster plot (plt.pcolormesh), where contours or margin patches can be added.
        Margin patches are e.g. used for the visualization of sensor calibration margins on an uncropped dataframe.
        The rendered plot is accessible via the 'figure' attribute. Internally only the plot axes will be rendered
        and updated. The dataframe will be supplied via the 'render_frame' method.

        Args:
            calibrationdata (CalibrationData): Instance that contains information of the current calibration values.
            contours (bool): Flag that enables or disables contours plotting.
                (default is True)
            margins (bool): Flag that enables or disables plotting of margin patches.
            vmin (float): ...
            vmax (float): ...
            cmap (str or plt.Colormap): Matplotlib colormap, given as name or instance.
            over (e.g. str): Color used for values above the expected data range.
            under (e.g. str): Color used for values below the expected data range.
            bad (e.g. str): Color used for invalid or masked data values.
            norm: Future feature!
            lot: Future feature!
            margin_color (e.g. str): Color of margin patches if enabled.
            margin_alpha (float): Transparency of margin patches.
                (default is 0.5)
            contours_step (int): Size of step between contour lines in model units.
                (default is 10)
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
        """
        self.calib = calibrationdata

        # flags
        self.margins = margins
        self.contours = contours
        self.show_lith = show_lith
        self.show_faults = show_faults
        #self.marker_position = marker_position
        self.minor_contours = minor_contours
        self.colormap = True

        # z-range handling
        if vmin is not None:
            self.vmin = vmin
        else:
            self.vmin = self.calib.s_min

        if vmax is not None:
            self.vmax = vmax
        else:
            self.vmax = self.calib.s_max

        self.model = model
        if self.model is not None:
            self.cmap = mcolors.ListedColormap(list(self.model.surfaces.df['color']))
        else:
        # pcolormesh setup
            self.cmap = plt.cm.get_cmap(cmap)
        if over is not None:
            self.cmap.set_over(over, 1.0)
        if under is not None:
            self.cmap.set_under(under, 1.0)
        if bad is not None:
            self.cmap.set_bad(bad, 1.0)

        self.norm = norm  # TODO: Future feature
        self.lot = lot  # TODO: Future feature


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

        # margin patches setup
        self.margin_color = margin_color
        self.margin_alpha = margin_alpha

        #aruco setup
        self.aruco_connect = aruco_connect
        self.aruco_scatter = aruco_scatter
        self.aruco_annotate = aruco_annotate
        self.aruco_color = aruco_color

        # TODO: save the figure's Matplotlib number to recall?
        # self.number = None
        self.figure = None
        self.ax = None
        self.create_empty_frame()

    def create_empty_frame(self):
        """ Initializes the matplotlib figure and empty axes according to projector calibration.

        The figure can be accessed by its attribute. It will be 'deactivated' to prevent random apperance in notebooks.
        """
        self.figure= Figure(figsize=(self.calib.p_frame_width / self.dpi, self.calib.p_frame_height / self.dpi),
                            dpi=self.dpi)
        #self.figure = plt.figure(figsize=(self.calib.p_frame_width / self.dpi, self.calib.p_frame_height / self.dpi),
        #                         dpi=self.dpi)
        self.ax = plt.Axes(self.figure, [0., 0., 1., 1.])
        self.figure.add_axes(self.ax)
        plt.close(self.figure)  # close figure to prevent inline display
        self.ax.set_axis_off()

    def render_frame(self, data, contourdata=None, vmin=None, vmax=None): #, df_position=None):  # ToDo: use keyword arguments
        """Renders a new frame according to class parameters.

        Resets the plot axes and redraws it with a new data frame, figure object remains untouched.
        If the data frame represents geological information (i.e. not topographical height), an optional data frame
        'contourdata' can be passed.

        Args:
            data (numpy.array): Current data frame representing surface height or geology
            contourdata (numpy.array): Current data frame representing surface height, if data is not height
                (default is None)
        """

        self.ax.cla()  # clear axes to draw new ones on figure
        if vmin is None:
            vmin = self.vmin
        if vmax is None:
            vmax = self.vmax

        if self.colormap:
            self.ax.pcolormesh(data, vmin=vmin, vmax=vmax, cmap=self.cmap, norm=self.norm)

        if self.contours:
            if contourdata is None:
                self.add_contours(data)
            else:
                self.add_contours(contourdata)

        if self.margins:
            self.add_margins()

    def add_margins(self):
        """ Adds margin patches to the current plot object.
        This is only useful when an uncropped dataframe is passed.
        """

        rec_t = plt.Rectangle((0, self.calib.s_height - self.calib.s_top), self.calib.s_width, self.calib.s_top,
                              fc=self.margin_color, alpha=self.margin_alpha)
        rec_r = plt.Rectangle((self.calib.s_width - self.calib.s_right, 0), self.calib.s_right, self.calib.s_height,
                              fc=self.margin_color, alpha=self.margin_alpha)
        rec_b = plt.Rectangle((0, 0), self.calib.s_width, self.calib.s_bottom,
                              fc=self.margin_color, alpha=self.margin_alpha)
        rec_l = plt.Rectangle((0, 0), self.calib.s_left, self.calib.s_height,
                              fc=self.margin_color, alpha=self.margin_alpha)

        self.ax.add_patch(rec_t)
        self.ax.add_patch(rec_r)
        self.ax.add_patch(rec_b)
        self.ax.add_patch(rec_l)

    @property
    def contours_levels(self):
        """Returns the current contour levels, being aware of changes in calibration."""

        return numpy.arange(self.vmin, self.vmax, self.contours_step)

    @property
    def contours_levels_minor(self):
        """Returns the current contour levels, being aware of changes in calibration."""

        return numpy.arange(self.vmin, self.vmax, self.contours_step_minor)

    def add_contours(self, data, extent=None):
        """Renders contours to the current plot object.
        Uses the different attributes to style contour lines and contour labels.
        """

        contours = self.ax.contour(data,
                                   levels=self.contours_levels,
                                   linewidths=self.contours_width,
                                   colors=self.contours_color,
                                   extent=extent)

        if self.minor_contours:
            self.ax.contour(data,
                            levels=self.contours_levels_minor,
                            linewidths=self.contours_width_minor,
                            colors=self.contours_color,
                            extent=extent)


        if self.contours_label:
            self.ax.clabel(contours,
                           inline=self.contours_label_inline,
                           fontsize=self.contours_label_fontsize,
                           fmt=self.contours_label_format)
                           #extent=extent)

    def update_model(self, model):
        self.model = model
        self.cmap = mcolors.ListedColormap(list(self.model.surfaces.df['color']))

    def add_faults(self):
        self.extract_boundaries(e_faults=True, e_lith=False)

    def add_lith(self):
        self.extract_boundaries(e_faults=False, e_lith=True)

    def extract_boundaries(self, e_faults=False, e_lith=False):
        faults = list(self.model._faults.df[self.model._faults.df['isFault'] == True].index)
        shape = self.model._grid.topography.resolution
        a = self.model.solutions.geological_map[1]
        extent = self.model._grid.topography.extent
        zorder = 2
        counter = a.shape[0]

        if e_faults:
            counters = numpy.arange(0, len(faults), 1)
            c_id = 0  # color id startpoint
        elif e_lith:
            counters = numpy.arange(len(faults), counter, 1)
            c_id = len(faults)  # color id startpoint
        else:
            raise AttributeError

        for f_id in counters:
            block = a[f_id]
            level = self.model.solutions.scalar_field_at_surface_points[f_id][numpy.where(
                self.model.solutions.scalar_field_at_surface_points[f_id] != 0)]

            levels = numpy.insert(level, 0, block.max())
            c_id2 = c_id + len(level)
            if f_id == counters.max():
                levels = numpy.insert(levels, level.shape[0], block.min())
                c_id2 = c_id + len(levels)  # color id endpoint
            block = block.reshape(shape)
            zorder = zorder - (f_id + len(level))

            if f_id >= len(faults):
                self.ax.contourf(block, 0, levels=numpy.sort(levels), colors=self.cmap.colors[c_id:c_id2][::-1],
                                 linestyles='solid', origin='lower',
                                 extent=extent, zorder=zorder)
            else:
                self.ax.contour(block, 0, levels=numpy.sort(levels), colors=self.cmap.colors[c_id:c_id2][0],
                                linestyles='solid', origin='lower',
                                extent=extent, zorder=zorder)
            c_id += len(level)

    def plot_aruco(self, df_position):
        if len(df_position) > 0:

            if self.aruco_scatter:
                self.ax.scatter(df_position[df_position['is_inside_box']]['box_x'].values,
                                df_position[df_position['is_inside_box']]['box_y'].values,
                                s=350, facecolors='none', edgecolors=self.aruco_color, linewidths=2)

                if self.aruco_annotate:
                    for i in range(len(df_position[df_position['is_inside_box']])):
                        self.ax.annotate(str(df_position[df_position['is_inside_box']].index[i]),
                                         (df_position[df_position['is_inside_box']]['box_x'].values[i],
                                         df_position[df_position['is_inside_box']]['box_y'].values[i]),
                                         c=self.aruco_color,
                                         fontsize=20,
                                         textcoords='offset pixels',
                                         xytext=(20, 20))

            if self.aruco_connect:
                self.ax.plot(df_position[df_position['is_inside_box']]['box_x'].values,
                             df_position[df_position['is_inside_box']]['box_y'].values,
                             linestyle='solid',
                             color=self.aruco_color)

            self.ax.set_axis_off()

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
        self._widget_plot_step_contours = pn.widgets.Spinner(name='Choose a contour step',value =self.contours_step )
        #self._widget_plot_step_contours = pn.widgets.TextInput(name='Choose a contour step')
        self._widget_plot_step_contours.param.watch(self._callback_plot_step_contours, 'value', onlychanged=False)
        #self._widget_plot_step_contours.value = str(self.contours_step)

        self._widget_plot_step_minorcontours = pn.widgets.Spinner(name='Choose a minor contour step', value = self.contours_step_minor)
        self._widget_plot_step_minorcontours.param.watch(self._callback_plot_step_minorcontours, 'value', onlychanged=False)
        #self._widget_plot_step_minorcontours.value = str(self.contours_step_minor)

        self._widget_plot_contours_label = pn.widgets.Checkbox(name='Show contours label', value=self.contours_label)
        self._widget_plot_contours_label.param.watch(self._callback_plot_contours_label, 'value',
                                                    onlychanged=False)

        self._widget_plot_contours_label_fontsize = pn.widgets.Spinner(name='set a contour label fontsize', value=self.contours_label_fontsize)
        self._widget_plot_contours_label_fontsize.param.watch(self._callback_plot_contours_label_fontsize, 'value', onlychanged=False)
        #self._widget_plot_contours_label_fontsize.value = str(self.contours_label_fontsize)

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

        self._widget_aruco_connect = pn.widgets.Checkbox(name='Show line connecting arucos', value=self.aruco_connect)
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

