from .aruco import ArucoMarkers
import panel as pn
import matplotlib.colors as mcolors
import numpy
import weakref


class MarkerDetection:
    def __init__(self, sensor, **kwargs):
        self.sensor = sensor
        self.Aruco = ArucoMarkers(sensor=sensor, **kwargs)
        self.df_aruco_position = None
        self.lines = None
        self.scat = None
        self._scat = None  # weak reference to a scat plot
        self._lin = None  # weak reference to a lines plot
        self.anot = None
        # aruco setup
        self.aruco_connect = False
        self.aruco_scatter = True
        self.aruco_annotate = True
        self.aruco_color = mcolors.to_hex('red')
        # Dummy sensor
        self._dict_position = {}
        self._widget_position = {}
        self._depth_frame = numpy.ones((sensor.extent[1], sensor.extent[3]))
        return print("Aruco detection ready")

    def update(self):
        if self.Aruco.kinect == "dummy":
            kwargs = {"dict_position": self._all_dict_position,
                      "depth_frame": self._depth_frame}
        else:
            kwargs = {}
        self.Aruco.search_aruco(**kwargs)
        self.Aruco.update_marker_dict()
        self.Aruco.transform_to_box_coordinates()
        self.df_aruco_position = self.Aruco.aruco_markers
        return self.df_aruco_position

    def set_aruco_position(self, dict_position: dict = {}, frame=None):
        """
        This function will create the aruco data frame with this values
        Args:
            dict_position: i.e. {1:[10,20],2:[100,200]} -> keys are the ids and the array the position in x and y
            frame: the depth frame, to extract the z value
        Returns:
        """
        self._dict_position = dict_position
        if frame is not None:
            self._depth_frame = frame

    def delete_aruco_position(self, ids = None):
        """If None, delete all positions. If ids match with one aruco only delete that one
        Args:
            ids: aruco id to delete
        """
        if ids:
            if ids in self.dict_position.keys():
                del self.dict_position[ids]
            else:
                print("id: ", ids, " not found")
        else:
            self._dict_position = {}

    def _set_widget_position(self, dict_position: dict = {}):
        self._widget_position = {**self._widget_position, **dict_position}

    @property
    def _all_dict_position(self):
        return {**self._dict_position, **self._widget_position}

    def calibrate_aruco(self, move_x, move_y):
        self.Aruco.set_xy_correction(move_x, move_y)

    def plot_aruco(self, ax, df_position=None):
        if self._scat is not None and self._scat() not in ax.collections:
            self.scat = None
        if self._lin is not None and self._lin() not in ax.lines:
            self.lines = None
        if len(df_position) > 0:
            df = df_position.loc[df_position.is_inside_box, ("box_x", "box_y")]
            if self.aruco_scatter:
                if self.scat is None:
                    self.scat = ax.scatter(df.box_x.values,
                                           df.box_y.values,
                                           s=350, facecolors='none', edgecolors=self.aruco_color, linewidths=2,
                                           zorder=20)
                    self._scat = weakref.ref(self.scat)

                else:
                    self.scat.set_offsets(numpy.c_[df.box_x.values, df.box_y.values])
                    self.scat.set_edgecolor(self.aruco_color)

                if self.aruco_annotate:
                    if self.anot is not None:
                        [ax.texts.remove(anot) for anot in self.anot if anot in ax.texts]
                        self.anot = None
                    self.anot = [ax.annotate(str(df.index[i]),
                                             (df.box_x.values[i], df.box_y.values[i]),
                                             c=self.aruco_color,
                                             fontsize=20,
                                             textcoords='offset pixels',
                                             xytext=(20, 20),
                                             zorder=21) for i in range(len(df))]

                else:
                    if self.anot is not None:
                        [ax.texts.remove(anot) for anot in self.anot if anot in ax.texts]
                        self.anot = None
            else:
                if self.scat is not None:
                    self.scat.remove()
                    self.scat = None

            if self.aruco_connect:
                if self.lines is None:
                    self.lines, = ax.plot(df_position[df_position['is_inside_box']]['box_x'].values,
                                          df_position[df_position['is_inside_box']]['box_y'].values,
                                          linestyle='solid',
                                          color=self.aruco_color,
                                          zorder=22)
                    self._lin = weakref.ref(self.lines)

                else:
                    self.lines.set_data(df_position[df_position['is_inside_box']]['box_x'].values,
                                        df_position[df_position['is_inside_box']]['box_y'].values)
                    self.lines.set_color(self.aruco_color)
            else:
                if self.lines is not None:
                    self.lines.remove()
                self.lines = None
        else:
            if self.lines is not None:
                self.lines.remove()
                self.lines = None
            if self.scat is not None:
                self.scat.remove()
                self.scat = None
            if self.anot is not None:
                [ax.texts.remove(anot) for anot in self.anot if anot in ax.texts]
                self.anot = None

        return ax

    # Widgets for aruco plotting

    def widgets_aruco(self):
        self._create_aruco_widgets()
        if self.Aruco.kinect == "dummy":
            pane = self._create_dummy_aruco()
        else:
            pane = None
        widgets = pn.WidgetBox(self._widget_aruco_scatter,
                               self._widget_aruco_annotate,
                               self._widget_aruco_connect,
                               self._widget_aruco_color)

        panel = pn.Column("<b> Dashboard for aruco Visualization </b>", pn.Row(widgets, pane))
        return panel

    def _create_aruco_widgets(self):
        self._widget_aruco_scatter = pn.widgets.Checkbox(name='Show aruco location', value=self.aruco_scatter)
        self._widget_aruco_scatter.param.watch(self._callback_aruco_scatter, 'value',
                                               onlychanged=False)

        self._widget_aruco_annotate = pn.widgets.Checkbox(name='Show aruco id', value=self.aruco_annotate)
        self._widget_aruco_annotate.param.watch(self._callback_aruco_annotate, 'value',
                                                onlychanged=False)

        self._widget_aruco_connect = pn.widgets.Checkbox(name='Show line connecting arucos',
                                                         value=self.aruco_connect)
        self._widget_aruco_connect.param.watch(self._callback_aruco_connect, 'value',
                                               onlychanged=False)

        self._widget_aruco_color = pn.widgets.ColorPicker(name='Color', value=self.aruco_color)
        self._widget_aruco_color.param.watch(self._callback_aruco_color, 'value', onlychanged=False)

    def _create_dummy_aruco(self):
        self._widget_aruco_id = pn.widgets.Spinner(name='id', value=1, step=1, width=70)
        self._widget_aruco_x = pn.widgets.Spinner(name="x coordinate", value=50, step=1, width=70)
        self._widget_aruco_y = pn.widgets.Spinner(name="y coordinate", value=50, step=1, width=70)

        self._widget_aruco_marker = pn.widgets.Button(name="Add dummy aruco dict", button_type="success")
        self._widget_aruco_marker.on_click(self._callback_set_aruco)

        self._widget_clear_marker = pn.widgets.Button(name="Clear aruco dict", button_type="warning")
        self._widget_clear_marker.on_click(self._callback_clear_aruco)

        pane = pn.WidgetBox("<b> Dummy aruco </b>",
                            pn.Row(self._widget_aruco_id, self._widget_aruco_x, self._widget_aruco_y),
                            self._widget_aruco_marker,
                            self._widget_clear_marker)
        return pane

    def _callback_aruco_scatter(self, event): self.aruco_scatter = event.new

    def _callback_aruco_annotate(self, event): self.aruco_annotate = event.new

    def _callback_aruco_connect(self, event): self.aruco_connect = event.new

    def _callback_aruco_color(self, event): self.aruco_color = event.new

    def _callback_clear_aruco(self, event): self._widget_position = {}

    def _callback_set_aruco(self, event):
        x = self._widget_aruco_x.value
        y = self._widget_aruco_y.value
        arucoid = self._widget_aruco_id.value
        self._set_widget_position(dict_position={arucoid: [x, y]})
