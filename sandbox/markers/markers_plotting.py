from .aruco import ArucoMarkers
import panel as pn
import matplotlib.colors as mcolors


class MarkerDetection: #TODO: include here the connection to the aruco markers
    def __init__(self, sensor):
        self.sensor = sensor
        self.Aruco = ArucoMarkers(sensor=sensor)
        self.df = None
        # aruco setup
        self.aruco_connect = True
        self.aruco_scatter = True
        self.aruco_annotate = True
        self.aruco_color = 'red'


    def update(self, **kwargs):
        self.Aruco.search_aruco(**kwargs)
        self.Aruco.update_marker_dict()
        self.Aruco.transform_to_box_coordinates()
        self.df = self.Aruco.aruco_markers
        return self.df

    def plot_aruco(self, ax, df_position=None):
        if df_position is None:
            df_position = self.update()
        if len(df_position) > 0:

            if self.aruco_scatter:
                ax.scatter(df_position[df_position['is_inside_box']]['box_x'].values,
                            df_position[df_position['is_inside_box']]['box_y'].values,
                            s=350, facecolors='none', edgecolors=self.aruco_color, linewidths=2)

                if self.aruco_annotate:
                    for i in range(len(df_position[df_position['is_inside_box']])):
                        ax.annotate(str(df_position[df_position['is_inside_box']].index[i]),
                                     (df_position[df_position['is_inside_box']]['box_x'].values[i],
                                     df_position[df_position['is_inside_box']]['box_y'].values[i]),
                                     c=self.aruco_color,
                                     fontsize=20,
                                     textcoords='offset pixels',
                                     xytext=(20, 20))

            if self.aruco_connect:
                ax.plot(df_position[df_position['is_inside_box']]['box_x'].values,
                         df_position[df_position['is_inside_box']]['box_y'].values,
                         linestyle='solid',
                         color=self.aruco_color)

            ax.set_axis_off()

        return ax

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



