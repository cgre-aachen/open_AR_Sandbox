import panel as pn
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

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

    def __init__(self, calibrationdata, use_panel=True):
        self.calib = calibrationdata

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

        pn.extension(raw_css=[self.css])
        # Create a panel object and serve it within an external bokeh browser that will be opened in a separate window

        # In this special case, a "tight" layout would actually add again white space to the plt canvas,
        # which was already cropped by specifying limits to the axis

        fig = Figure()
        self.frame = pn.pane.Matplotlib(fig,
                                        width=self.calib.p_frame_width,
                                        height=self.calib.p_frame_height,
                                        margin=[self.calib.p_frame_top, 0, 0, self.calib.p_frame_left],
                                        tight=False,
                                        dpi=self.dpi,
                                        css_classes=['frame']
                                        )
        # ax = fig.subplots()
        # ax.plot([1, 2, 3])
        # panel_fig = pn.pane.Matplotlib(fig)

        plt.close()  # close figure to prevent inline display

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
                                 margin=[self.calib.p_frame_top, 0, 0, 0],
                                 )

        self.panel = pn.Row(self.frame, self.sidebar,
                            width=self.calib.p_width,
                            height=self.calib.p_height,
                            sizing_mode='fixed',
                            css_classes=['panel']
                            )
        return True

    def start_server(self):
        # TODO: Add specific port? port=4242
        # Check for instances and close them?
        self.panel.show(threaded=True)#, title="Sandbox frame!")#, port = 4242, use_reloader = False)
        print('Projector initialized and server started.\n'
              'Please position the browser window accordingly and enter fullscreen!')
        #self.server = pn.serve(self.panel, port=0, title="Sandbox frame!")

        return True

    def show(self, figure):
        self.frame.object = figure
        return True

    def trigger(self):
        self.frame.param.trigger('object')
        return True
