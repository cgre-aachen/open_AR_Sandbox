import panel as pn
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import json


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

        self.ax = plt.Axes(self.figure, [0., 0., 1., 1.])
        self.figure.add_axes(self.ax)
        self.ax.set_axis_off()

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
        return True

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


