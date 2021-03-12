import os
import panel as pn
import matplotlib
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import json
from sandbox import _calibration_dir


class Projector(object):
    dpi = 100  # make sure that figures can be displayed pixel-precise

    css = '''
    body {
      margin:0px;
      background-color: #FFFFFF;
    }
    .panel {
      background-color: #000000;
      overflow: hidden;
    }
    .bk.frame {
      background-color: #FFFFFF;
      color: #FFFFFF;
    }
    .bk.legend {
      background-color: #16425B;
      color: #CCCCCC;
    }
    .bk.hot {
      background-color: #2896A5;
      color: #CCCCCC;
    }
    .bk.profile {
      background-color: #40C1C7;
      color: #CCCCCC;
    }
    .bk.colorbar {
      background-color: #2896A5;
      color: #CCCCCC;
    '''

    # F5FCFF
    def __init__(self, calibprojector: str = None, use_panel: bool = True, p_width=1280, p_height=800,
                 show_colorbar: bool = False, show_legend: bool = False, show_hot: bool = False,
                 show_profile: bool = False, position_colorbar: str = "vertical"):
        """
        Args:
            calibprojector:
            use_panel: Automatically display
            p_width: x native resolution of the projector
            p_height: y native resolution of the projector
            show_colorbar:
            show_legend:
            show_hot:
            show_profile
        """
        self.version = '2.0.p'
        self.ax = None
        self.figure = None
        self.json_filename = calibprojector
        
        if calibprojector is None:
            self.p_width = p_width
            self.p_height = p_height
            self.p_frame_top = 50
            self.p_frame_left = 50
            self.p_frame_width = 700
            self.p_frame_height = 500
        else:
            self.load_json(calibprojector)

        # flags
        self.enable_legend = show_legend
        self.enable_hot = show_hot
        self.enable_colorbar = show_colorbar
        self.pos_colorbar = position_colorbar
        self._ratio = 10
        self.enable_profile = show_profile
        self._dim_label_ax = [0, 0, 2, 0.1] if self.pos_colorbar == "horizontal" else [0, 0, 0.1, 2]
        self._size_label_cbar = 15

        # panel components (panes)
        self.panel = None
        self.frame = None
        self.legend = None
        self.hot = None
        self.profile = None
        self.colorbar = None
        self.sidebar = None
        # This is to solve issue #3. Give 0.01 ms to each Text from ax.arists to be plotted
        self._target_time = 0.01
        self._paused_time = None

        self.create_panel()
        if use_panel is True:
            self.start_server()

    def create_panel(self):
        """ Initializes the matplotlib figure and empty axes according to projector calibration.

        The figure can be accessed by its attribute. It will be 'deactivated' to prevent random apperance in notebooks.
        """
        pn.extension(raw_css=[self.css])
        # Create a panel object and serve it within an external bokeh browser that will be opened in a separate window

        # In this special case, a "tight" layout would actually add again white space to the plt canvas,
        # which was already cropped by specifying limits to the axis

        self.figure = Figure(figsize=(self.p_frame_width / self.dpi, self.p_frame_height / self.dpi),
                             dpi=self.dpi)
        self.ax = Axes(self.figure, [0., 0., 1., 1.])
        self.figure.add_axes(self.ax)
        self.ax.set_axis_off()
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)

        self.frame = pn.pane.Matplotlib(self.figure,
                                        width=self.p_frame_width,
                                        height=self.p_frame_height,
                                        margin=[self.p_frame_top, 0, 0, self.p_frame_left],
                                        tight=False,
                                        dpi=self.dpi,
                                        css_classes=['frame']
                                        )
        plt.close(self.figure)  # close figure to prevent inline display

        if self.enable_colorbar:
            empty_fig_bg_cb = Figure()
            self.colorbar = pn.pane.Matplotlib(empty_fig_bg_cb,
                                               width=self.p_frame_width if self.pos_colorbar == "horizontal"
                                               else round(self.p_frame_width/self._ratio),
                                               height=self.p_frame_height if self.pos_colorbar == "vertical"
                                               else round(self.p_frame_height/self._ratio),
                                               margin=[0,0,0,0],
                                               dpi=self.dpi,
                                               css_classes=['colorbar'],
                                               tight=True)

        if self.enable_legend:
            self.legend = pn.Column("### Legend",
                                    # add parameters from calibration for positioning
                                    width=100,
                                    height=100,
                                    margin=[0, 0, 0, 0],
                                    css_classes=['legend'])

        if self.enable_hot:
            self.hot = pn.Column("### Hot area",
                                 width=100,
                                 height=100,
                                 margin=[0, 0, 0, 0],
                                 css_classes=['hot']
                                 )

        if self.enable_profile:
            self.profile = pn.Column("### Profile",
                                     width=100,
                                     height=100,
                                     margin=[0, 0, 0, 0],
                                     css_classes=['profile']
                                     )

        # Combine panel and deploy bokeh server
        if self.pos_colorbar == "vertical":
            self.sidebar = pn.Column(self.colorbar, self.legend, self.hot, self.profile,
                                     margin=[self.p_frame_top, 0, 0, 0],
                                     )

            self.panel = pn.Row(self.frame, self.sidebar,
                                width=self.p_width,
                                height=self.p_height,
                                sizing_mode='fixed',
                                css_classes=['panel']
                                )
        elif self.pos_colorbar == "horizontal":
            self.sidebar = pn.Column(self.legend, self.hot, self.profile,
                                     margin=[self.p_frame_top, 0, 0, 0],
                                     )
            self.colorbar.margin = [0, 0, 0, self.p_frame_left]
            self.panel = pn.Row(pn.Column(self.frame, self.colorbar),
                                self.sidebar,
                                width=self.p_width,
                                height=self.p_height,
                                sizing_mode='fixed',
                                css_classes=['panel']
                                )
        else:
            raise AttributeError

        return True

    def set_colorbar(self, vmin, vmax, cmap, norm=None, label=None):
        """

        Args:
            vmin:
            vmax:
            cmap:
            norm:
            label:

        Returns:

        """
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        cb = Figure()
        ax = Axes(cb, self._dim_label_ax)
        cb.add_axes(ax)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax) if norm is None else norm
        cb1 = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation=self.pos_colorbar)
        cb1.set_label(label, size=self._size_label_cbar) if label is not None else None
        cb1.ax.tick_params(labelsize=self._size_label_cbar)
        self.colorbar.object = cb
        self.colorbar.param.trigger("object")

    def write_text(self, text: str = "cgre-aachen / open_AR_Sandbox"):
        """
        Display a custom text to be displayed in the middle of the sandbox
        Args:
            text: message to display
        Returns:

        """
        self.ax.texts = []
        x = (self.ax.get_xlim()[1] - self.ax.get_xlim()[0])/2
        y = (self.ax.get_ylim()[1] - self.ax.get_ylim()[0])/2
        self.ax.annotate(text, (x, y), zorder=1000, xycoords="data", fontsize=18, ha='center',
                         va='top', wrap=True)
        self.trigger()

    def _replace_figure_with_pyplot(self):
        """Deprecated!! workaround to fix bug of no dpi"""
        figure = plt.figure(figsize=(self.p_frame_width / self.dpi, self.p_frame_height / self.dpi),
                            dpi=self.dpi)
        ax = plt.Axes(figure, [0., 0., 1., 1.])
        figure.add_axes(ax)
        plt.close(figure)  # close figure to prevent inline display
        ax.set_axis_off()
        self.figure = figure
        self.ax = ax
        self.frame.object = figure
        self.trigger()

    def start_server(self):
        """
        Display the panel object in a new browser window
        Returns:

        """
        # Check for instances and close them?
        self.panel.show(threaded=False)  # , title="Sandbox frame!")#, port = 4242, use_reloader = False)
        # TODO: check how can check if the port exist/open and overwrite it
        print('Projector initialized and server started.\n'
              'Please position the browser window accordingly and enter fullscreen!')
        return True

    def clear_axes(self):
        """
        Empty the axes of the current figure and trigger the update of the figure in the panel.
        Returns:

        """
        self.ax.cla()
        self.trigger()
        return True

    def _clock(self):
        """
        To to be sure that panel have enough time to display the figure he want. Solves issue #3
        """
        ctext = [isinstance(text, matplotlib.text.Text) for text in self.ax.artists]
        if True in ctext:
            sec = len(ctext)*self._target_time  # Give 0.005 ms to each Text from contours to be plotted
            self._paused_time = sec
            plt.pause(sec)
        else:
            self._paused_time = None

    def trigger(self):
        """
        Update the panel figure if modified 
        Returns:

        """

        # self.figure.canvas.draw_idle()  # TODO: do we need this or not?
        self.frame.param.trigger('object')
        self._clock()
        return True

    def save_json(self, file: str = 'projector_calibration.json'):
        """
        Saves the current state of the projector in a .JSON calibration file
        Args:
            file: address to save the calibration

        Returns:

        """
        with open(file, "w") as calibration_json:
            data = {'version': self.version,
                    'p_width': self.p_width,
                    'p_height': self.p_height,
                    'p_frame_top': self.p_frame_top,
                    'p_frame_left': self.p_frame_left,
                    'p_frame_width': self.p_frame_width,
                    'p_frame_height': self.p_frame_height}
            json.dump(data, calibration_json)
        print('JSON configuration file saved:', str(file))
        return True

    def load_json(self, file: str):
        """
        Load a calibration file (.JSON format) and actualizes the panel parameters
        Args:
            file: address of the calibration to load

        Returns:

        """
        def json_load(dict_data):
            if dict_data['version'] == self.version:
                self.p_width = dict_data['p_width']
                self.p_height = dict_data['p_height']
                self.p_frame_top = dict_data['p_frame_top']
                self.p_frame_left = dict_data['p_frame_left']
                self.p_frame_width = dict_data['p_frame_width']
                self.p_frame_height = dict_data['p_frame_height']
                print("JSON configuration loaded for projector.")
            else:
                print("JSON configuration incompatible." +
                      "\nPlease select a valid calibration file or start a new calibration!")
        if os.path.isfile(file):
            with open(file) as calibration_json:
                data = json.load(calibration_json)
                json_load(data)
        else:
            data = json.loads(file)
            json_load(data)
        return True

    def calibrate_projector(self):
        self._create_widgets()
        panel = pn.Column("### Projector dashboard arrangement",
                          self._widget_p_frame_top,
                          self._widget_p_frame_left,
                          self._widget_p_frame_width,
                          self._widget_p_frame_height,
                          '<b>Save file<b>',
                          self._widget_json_filename,
                          self._widget_json_save
                          )
        return panel

    def _create_widgets(self):
        # projector widgets and links
        self._widget_p_frame_top = pn.widgets.IntSlider(name='Main frame top margin',
                                                        value=self.p_frame_top,
                                                        start=0,
                                                        end=self.p_height - 20)
        self._widget_p_frame_top.link(self.frame, callbacks={'value': self._callback_p_frame_top})

        self._widget_p_frame_left = pn.widgets.IntSlider(name='Main frame left margin',
                                                         value=self.p_frame_left,
                                                         start=0,
                                                         end=self.p_width - 20)
        self._widget_p_frame_left.link(self.frame, callbacks={'value': self._callback_p_frame_left})

        self._widget_p_frame_width = pn.widgets.IntSlider(name='Main frame width',
                                                          value=self.p_frame_width,
                                                          start=10,
                                                          end=self.p_width)
        self._widget_p_frame_width.link(self.frame, callbacks={'value': self._callback_p_frame_width})

        self._widget_p_frame_height = pn.widgets.IntSlider(name='Main frame height',
                                                           value=self.p_frame_height,
                                                           start=10,
                                                           end=self.p_height)
        self._widget_p_frame_height.link(self.frame, callbacks={'value': self._callback_p_frame_height})

        self._widget_json_filename = pn.widgets.TextInput(name='Choose a calibration filename:')
        self._widget_json_filename.param.watch(self._callback_json_filename, 'value', onlychanged=False)
        self._widget_json_filename.value = _calibration_dir + 'my_projector_calibration.json'

        self._widget_json_save = pn.widgets.Button(name='Save calibration')
        self._widget_json_save.param.watch(self._callback_json_save, 'clicks', onlychanged=False)

        return True

    def _callback_p_frame_top(self, target, event):
        self.p_frame_top = event.new
        m = target.margin
        n = event.new
        # just changing single indices does not trigger updating of pane
        target.margin = [n, m[1], m[2], m[3]]

    def _callback_p_frame_left(self, target, event):
        self.p_frame_left = event.new
        m = target.margin
        n = event.new
        target.margin = [m[0], m[1], m[2], n]

    def _callback_p_frame_width(self, target, event):
        self.p_frame_width = event.new
        target.width = event.new
        target.param.trigger('object')

    def _callback_p_frame_height(self, target, event):
        self.p_frame_height = event.new
        target.height = event.new
        target.param.trigger('object')

    def _callback_json_filename(self, event):
        self.json_filename = event.new

    def _callback_json_save(self, event):
        if self.json_filename is not None:
            self.save_json(file=self.json_filename)
