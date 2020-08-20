import panel as pn
pn.extension()
from matplotlib.figure import Figure
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
    '''

    def __init__(self, calibprojector: str = None, use_panel: bool = True):
        """

        Args:
            calibprojector:
            use_panel:
        """
        self.version = '2.0.p'
        self.ax = None
        self.figure = None
        self.json_filename = calibprojector
        
        if calibprojector is None:
            self.p_width = 1280
            self.p_height = 800
            self.p_frame_top = 50
            self.p_frame_left = 50
            self.p_frame_width = 700
            self.p_frame_height = 500
        else:
            self.load_json(calibprojector)

        # flags
        self.enable_legend = False
        self.enable_hot = False
        self.enable_profile = False

        # panel components (panes)
        self.panel = None
        self.frame = None
        self.legend = None
        self.hot = None
        self.profile = None
        self.sidebar = None
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
        #self.figure = plt.figure(figsize=(self.p_frame_width / self.dpi, self.p_frame_height / self.dpi),
        #                     dpi=self.dpi)
        #self.ax = plt.gca()
        self.ax = plt.Axes(self.figure, [0., 0., 1., 1.])
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
        self.sidebar = pn.Column(self.legend, self.hot, self.profile,
                                 margin=[self.p_frame_top, 0, 0, 0],
                                 )

        self.panel = pn.Row(self.frame, self.sidebar,
                            width=self.p_width,
                            height=self.p_height,
                            sizing_mode='fixed',
                            css_classes=['panel']
                            )
        #TODO: Super-dirty fix
        #self._replace_figure_with_pyplot()
        #self._paint_logo()

        return True

    def _paint_logo(self):
        self.ax.texts=[]
        self.ax.annotate("cgre-aachen / open_AR_Sandbox ", (self.p_frame_width/2, self.p_frame_height/2),
                         zorder=1)#c='k', fontsize=30)#, textcoords='offset pixels', xytext=(20, 20), zorder=21)
        self.trigger()

    def _replace_figure_with_pyplot(self):
        """workaround to fix bug of no dpi"""
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
        self.panel.show(threaded=True)#, title="Sandbox frame!")#, port = 4242, use_reloader = False)
        #TODO: check how can check if the port exist/open and overwrite it
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

    def trigger(self):
        """
        Update the panel figure if modified 
        Returns:

        """
        #self.figure.canvas.draw()
        #self.figure.canvas.flush_events()
        self.frame.param.trigger('object')
        return True

    def load_json(self, file: str):
       """
        Load a calibration file (.JSON format) and actualizes the panel parameters 
        Args:
            file: address of the calibration to load

        Returns:

        """
       with open(file) as calibration_json:
            data = json.load(calibration_json)
            if data['version'] == self.version:
                self.p_width = data['p_width']
                self.p_height = data['p_height']
                self.p_frame_top = data['p_frame_top']
                self.p_frame_left = data['p_frame_left']
                self.p_frame_width = data['p_frame_width']
                self.p_frame_height = data['p_frame_height']
                print("JSON configuration loaded for projector.")
            else:
                print("JSON configuration incompatible.\nPlease select a valid calibration file or start a new calibration!")

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

    def calibrate_projector(self):
        self._create_widgets()
        panel = pn.Column("### Projector dashboard arrangement",
                           self._widget_p_frame_top,
                           self._widget_p_frame_left,
                           self._widget_p_frame_width,
                           self._widget_p_frame_height,
                           #self._widget_p_enable_auto_calibration,
                           #self._widget_p_automatic_calibration,)
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

        # Auto- Calibration widgets

        #self._widget_p_enable_auto_calibration = pn.widgets.Checkbox(name='Enable Automatic Calibration', value=False)
        #self._widget_p_enable_auto_calibration.param.watch(self._callback_enable_auto_calibration, 'value',
        #                                                   onlychanged=False)

        #self._widget_p_automatic_calibration = pn.widgets.Button(name="Run", button_type="success")
        #self._widget_p_automatic_calibration.param.watch(self._callback_automatic_calibration, 'clicks',
        #                                                 onlychanged=False)

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

