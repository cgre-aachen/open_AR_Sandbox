from .aruco import ArucoMarkers
import panel as pn
import matplotlib.colors as mcolors


class MarkerDetection: #TODO: include here the connection to the aruco markers
    def __init__(self):
        pass

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



