from .aruco import ArucoMarkers
import panel as pn
import matplotlib.colors as mcolors
import numpy
import weakref

class MarkerDetection: #TODO: include here the connection to the aruco markers
    def __init__(self, sensor, **kwargs):
        self.sensor = sensor
        self.Aruco = ArucoMarkers(sensor=sensor, **kwargs)
        self.df = None
        self.lines = None
        self.scat = None
        self._scat = None # weak reference to a scat plot
        self._lin = None  # weak reference to a lines plot
        # aruco setup
        self.aruco_connect = True
        self.aruco_scatter = True
        self.aruco_annotate = True
        self.aruco_color = 'red'
        print("Aruco detection ready")


    def update(self, **kwargs):
        self.Aruco.search_aruco(**kwargs)
        self.Aruco.update_marker_dict()
        self.Aruco.transform_to_box_coordinates()
        self.df = self.Aruco.aruco_markers
        return self.df

    def plot_aruco(self, ax, df_position=None):
        ax.texts = []
        if self._scat is not None and self._scat() not in ax.collections:
            self.scat = None
        if self._lin is not None and self._lin() not in ax.lines:
            self.lines = None
        if len(df_position) > 0:
            if self.aruco_scatter:
                if self.scat is None:
                    self.scat = ax.scatter(df_position[df_position['is_inside_box']]['box_x'].values,
                                           df_position[df_position['is_inside_box']]['box_y'].values,
                                           s=350, facecolors='none', edgecolors=self.aruco_color, linewidths=2,
                                           zorder=20)
                    self._scat = weakref.ref(self.scat)

                else:
                    self.scat.set_offsets(numpy.c_[df_position[df_position['is_inside_box']]['box_x'].values,
                                                   df_position[df_position['is_inside_box']]['box_y'].values])
                    self.scat.set_edgecolor(self.aruco_color)

                if self.aruco_annotate:
                    for i in range(len(df_position[df_position['is_inside_box']])):
                        ax.annotate(str(df_position[df_position['is_inside_box']].index[i]),
                                     (df_position[df_position['is_inside_box']]['box_x'].values[i],
                                     df_position[df_position['is_inside_box']]['box_y'].values[i]),
                                     c=self.aruco_color,
                                     fontsize=20,
                                     textcoords='offset pixels',
                                     xytext=(20, 20),
                                    zorder=21)
                else: ax.texts = []
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
                                          zorder = 22)
                    self._lin = weakref.ref(self.lines)

                else:
                    self.lines.set_data(df_position[df_position['is_inside_box']]['box_x'].values,
                             df_position[df_position['is_inside_box']]['box_y'].values)
                    self.lines.set_color(self.aruco_color)
            else:
                if self.lines is not None: self.lines.remove()
                self.lines = None

            #ax.set_axis_off()
        else:
            if self.lines is not None:
                self.lines.remove()
                self.lines = None
            if self.scat is not None:
                self.scat.remove()
                self.scat = None
            ax.texts = []

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



