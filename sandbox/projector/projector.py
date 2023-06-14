import os
import panel as pn
import matplotlib
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import json
from sandbox import _calibration_dir, set_logger
logger = set_logger(__name__)


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

    def __init__(self, calibprojector: str = None, use_panel: bool = True, p_width=1280, p_height=800,
                 show_colorbar: bool = False, position_colorbar: str = "vertical",
                 show_legend: bool = False, show_hot: bool = False,
                 show_profile: bool = False, ):
        """
        Args:
            calibprojector:
            use_panel: Automatically display
            p_width: x native resolution of the projector
            p_height: y native resolution of the projector
            show_colorbar:
            position_colorbar: "vertical" or "horizontal"
            show_legend:
            show_hot:
            show_profile
        """
        self.version = '2.2.p'
        self.ax = None
        self.figure = None
        self.json_filename = calibprojector

        # flags
        self.enable_legend = show_legend
        self.enable_hot = show_hot
        self.enable_colorbar = show_colorbar
        self.pos_colorbar = position_colorbar
        self._ratio = 10
        self.enable_profile = show_profile

        if calibprojector is None:
            self.p_width = p_width
            self.p_height = p_height
            self.p_frame_top = 50
            self.p_frame_left = 50
            self.p_frame_width = 700
            self.p_frame_height = 500
            # Colorbar
            self.col_top = 0
            self.col_left = 0 if self.pos_colorbar == "vertical" else self.p_frame_left
            self.col_width = self.p_frame_width if self.pos_colorbar == "horizontal" \
                                           else round(self.p_frame_width / self._ratio)
            self.col_height = self.p_frame_height if self.pos_colorbar == "vertical" \
                                           else round(self.p_frame_height / self._ratio)

            self.leg_width = round(self.p_frame_width/4)
            self.leg_height = round(self.p_frame_width/3)
            self.leg_top = 0
            self.leg_left = 0
        else:
            self.load_json(calibprojector)

        self._size_label_cbar = 15
        self._label = None

        # panel components (panes)
        self.panel = None
        self.frame = None
        self.legend = None
        self.hot = None
        self.profile = None
        self.colorbar = None
        self.sidebar = None
        # This is to solve issue #3. Give 0.01 ms to each Text from ax.arists to be plotted
        self._target_time = 0.00
        self._paused_time = None

        self.create_panel()
        if use_panel is True:
            self.start_server()

    @property
    def _dim_label_ax(self):
        return [0, 0, 2, 0.1] if self.pos_colorbar == "horizontal" else [0, 0, 0.1, 2]

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
                                        margin=(self.p_frame_top, 0, 0, self.p_frame_left),
                                        tight=False,
                                        dpi=self.dpi,
                                        css_classes=['frame']
                                        )
        plt.close(self.figure)  # close figure to prevent inline display

        if self.enable_colorbar:
            self.create_colorbar()

        if self.enable_legend:
            self.create_legend()

        if self.enable_hot:
            self.create_hot()

        if self.enable_profile:
            self.create_profile()

        # Combine panel and deploy bokeh server
        if self.pos_colorbar == "vertical":
            self.sidebar = pn.Column(self.colorbar, self.legend, self.hot, self.profile,
                                     margin=(self.p_frame_top, 0, 0, 0),
                                     )

            self.panel = pn.Row(pn.Column(self.frame, None),
                                self.sidebar,
                                width=self.p_width,
                                height=self.p_height,
                                sizing_mode='fixed',
                                css_classes=['panel']
                                )
        elif self.pos_colorbar == "horizontal":
            self.sidebar = pn.Column(self.legend, self.hot, self.profile,
                                     margin=(self.p_frame_top, 0, 0, 0),
                                     )
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

    def create_colorbar(self):
        empty_fig_bg_cb = Figure()
        self.colorbar = pn.pane.Matplotlib(empty_fig_bg_cb,
                                           width= self.col_width,
                                           height= self.col_height,
                                           margin=(self.col_top, 0, 0, self.col_left),
                                           dpi=self.dpi*2,
                                           css_classes=['colorbar'],
                                           tight=True)

    def create_legend(self):
        empty_fig_bg_ld = Figure()
        self.legend = pn.pane.Matplotlib(empty_fig_bg_ld,
                                           width=self.leg_width,
                                           height=self.leg_height,
                                           margin=(self.leg_top, 0, 0, self.leg_left),
                                           dpi=self.dpi*2,
                                           css_classes=['legend'],
                                           tight=True)

    def create_hot(self):
        self.hot = pn.Column("### Hot area",
                             width=100,
                             height=100,
                             margin=(0, 0, 0, 0),
                             css_classes=['hot']
                             )

    def create_profile(self):
        self.profile = pn.Column("### Profile",
                                 width=100,
                                 height=100,
                                 margin=(0, 0, 0, 0),
                                 css_classes=['profile']
                                 )

    def set_colorbar(self, vmin: float, vmax: float, cmap="viridis", norm=None, label: str = None):
        """
        Create a colorbar and display the figure in the colorbar widget
        Args:
            vmin: Minimun value of the colorbar
            vmax: Maximum value of the colorbar
            cmap: Colormap of the colorbar
            norm: (optionl) Normalization, in case that not, this is internally managed
            label: Text to display as label in the colorbar
        Returns:

        """
        if self.colorbar is not None:
            if isinstance(cmap, str):
                cmap = plt.get_cmap(cmap)
            label = label if label is not None else self._label
            cb = Figure()
            ax = Axes(cb, self._dim_label_ax)
            cb.add_axes(ax)
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax) if norm is None else norm
            cb1 = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation=self.pos_colorbar)
            cb1.set_label(label, size=self._size_label_cbar) if label is not None else None
            cb1.ax.tick_params(labelsize=self._size_label_cbar)
            self.colorbar.object = cb
            self.colorbar.param.trigger("object")

    def set_legend(self, handles=None, labels=None, *args):
        """
        Create a legend with the information of frame with the ax.get_legend_handles_labels().
        External handles and labels can be used
        Returns:

        """
        if self.legend is not None:
            ld = Figure()
            if handles is None and labels is None:
                if args == ():
                    ld.legend(*self.ax.get_legend_handles_labels())
                else:
                    ld.legend(*args)
            else:
                ld.legend(labels=labels,
                          handles=handles
                          )
            self.legend.object = ld
            self.legend.param.trigger("object")


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
        self.ax.annotate(text, (x, y), zorder=1000000, xycoords="data", fontsize=18, ha='center',
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
        logger.info('Projector initialized and server started.\n'
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
        ctext = [isinstance(text, matplotlib.text.Text) for text in self.ax.texts]
        coll = [isinstance(coll, matplotlib.collections.PathCollection) for coll in self.ax.collections]
        if True in ctext or True in coll:
            sec = (len(coll)+len(ctext))*self._target_time  # Give 0.005 ms to each Text from contours to be plotted
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
                    'p_frame_height': self.p_frame_height,
                    'col_top': self.col_top,
                    'col_left': self.col_left,
                    'col_width': self.col_width,
                    'col_height': self.col_height,
                    'pos_colorbar': self.pos_colorbar,
                    'leg_top': self.leg_top,
                    'leg_left': self.leg_left,
                    'leg_width': self.leg_width,
                    'leg_height': self.leg_height,
                    }
            json.dump(data, calibration_json)
        logger.info('JSON configuration file saved: %s' % str(file))
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
                self.p_width = dict_data.get('p_width')
                self.p_height = dict_data.get('p_height')
                self.p_frame_top = dict_data.get('p_frame_top')
                self.p_frame_left = dict_data.get('p_frame_left')
                self.p_frame_width = dict_data.get('p_frame_width')
                self.p_frame_height = dict_data.get('p_frame_height')
                self.col_top = dict_data.get('col_top')
                self.col_left = dict_data.get('col_left')
                self.col_width = dict_data.get('col_width')
                self.col_height = dict_data.get('col_height')
                self.pos_colorbar = dict_data.get('pos_colorbar')
                self.leg_top = dict_data.get('leg_top')
                self.leg_left = dict_data.get('leg_left')
                self.leg_width = dict_data.get('leg_width')
                self.leg_height =dict_data.get('leg_height')

                logger.info("JSON configuration loaded for projector")
            else:
                logger.warning("JSON configuration incompatible." +
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

    def show_widgets_sidepanels(self):
        tabs = pn.Tabs(("Colorbar", self.show_widget_colorbar()),
                       ("Legend", self.show_widget_legend()))
        return tabs

    def show_widget_colorbar(self):
        self._create_widgets_colorbar()
        panel1 = pn.Column("### Colorbar",
                           self._widgets_show_colorbar,
                           self._widget_label,
                           self._widget_refresh_col
                          )
        panel2 = pn.Column(self._widget_colorbar_ori,
                           self._widget_top_colorbar,
                           self._widget_left_colorbar,
                           self._widget_width_colorbar,
                           self._widget_height_colorbar,
                           self._widget_col_background)
        return pn.Row(panel1, panel2)

    def show_widget_legend(self):
        self._create_widgets_legend()
        panel3 = pn.Column("### Legend",
                           self._widgets_show_legend,
                           self._widget_refresh_leg)
        panel4 = pn.Column(self._widget_top_legend,
                           self._widget_left_legend,
                           self._widget_width_legend,
                           self._widget_height_legend,
                           self._widget_leg_background)
        return pn.Row(panel3, panel4)

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

    def _create_widgets_colorbar(self):
        self._widget_colorbar_ori = pn.widgets.Select(name='Orientation Colorbar',
                                                    options=["vertical", "horizontal"],
                                                   value=self.pos_colorbar)
        self._widget_colorbar_ori.param.watch(self._callback_colorbar_ori, 'value', onlychanged=False)

        self._widgets_show_colorbar = pn.widgets.Checkbox(name='Show colorbar',
                                                          value=self.enable_colorbar)
        self._widgets_show_colorbar.param.watch(self._callback_enable_colorbar, 'value',
                                                onlychanged=False)

        self._widget_top_colorbar = pn.widgets.IntSlider(name='Top space',
                                                        value=self.col_top,
                                                        start=0,
                                                        end=self.p_height - 20)
        self._widget_top_colorbar.param.watch(self._callback_top_colorbar, 'value', onlychanged=False)

        self._widget_left_colorbar = pn.widgets.IntSlider(name='Left space',
                                                         value=self.col_left,
                                                         start=0,
                                                         end=self.p_width - 20)
        self._widget_left_colorbar.param.watch(self._callback_left_colorbar, 'value', onlychanged=False)

        self._widget_width_colorbar = pn.widgets.IntSlider(name='Width Colorbar',
                                                          value=self.col_width,
                                                          start=1,
                                                          end=self.p_width)
        self._widget_width_colorbar.param.watch(self._callback_width_colorbar, 'value', onlychanged=False)

        self._widget_height_colorbar = pn.widgets.IntSlider(name='Height colorbar',
                                                           value=self.col_height,
                                                           start=1,
                                                           end=self.p_height)
        self._widget_height_colorbar.param.watch(self._callback_height_colorbar, 'value', onlychanged=False)

        self._widget_label = pn.widgets.TextInput(name='Label of colorbar')
        self._widget_label.param.watch(self._callback_label, 'value', onlychanged=False)

        self._widget_refresh_col = pn.widgets.Button(name="Refresh label",
                                                    button_type="success")
        self._widget_refresh_col.param.watch(self._callback_refresh, 'clicks',
                                            onlychanged=False)

        self._widget_col_background = pn.widgets.ColorPicker(name='Color background colorbar', value="#2896A5")
        self._widget_col_background.param.watch(self._callback_col_background, 'value', onlychanged=False)

    def _create_widgets_legend(self):
        self._widgets_show_legend = pn.widgets.Checkbox(name='Show legend',
                                                        value=self.enable_legend)
        self._widgets_show_legend.param.watch(self._callback_enable_legend, 'value',
                                                onlychanged=False)

        self._widget_top_legend = pn.widgets.IntSlider(name='Top space',
                                                        value=self.leg_top,
                                                        start=0,
                                                        end=self.p_height - 20)
        self._widget_top_legend.param.watch(self._callback_top_legend, 'value', onlychanged=False)

        self._widget_left_legend = pn.widgets.IntSlider(name='Left space',
                                                         value=self.leg_left,
                                                         start=0,
                                                         end=self.p_width - 20)
        self._widget_left_legend.param.watch(self._callback_left_legend, 'value', onlychanged=False)

        self._widget_width_legend = pn.widgets.IntSlider(name='Width Legend',
                                                          value=self.leg_width,
                                                          start=1,
                                                          end=self.p_width)
        self._widget_width_legend.param.watch(self._callback_width_legend, 'value', onlychanged=False)

        self._widget_height_legend = pn.widgets.IntSlider(name='Height Legend',
                                                           value=self.leg_height,
                                                           start=1,
                                                           end=self.p_height)
        self._widget_height_legend.param.watch(self._callback_height_legend, 'value', onlychanged=False)

        self._widget_refresh_leg = pn.widgets.Button(name="Refresh legend",
                                                    button_type="success")
        self._widget_refresh_leg.param.watch(self._callback_refresh_leg, 'clicks',
                                            onlychanged=False)

        self._widget_leg_background = pn.widgets.ColorPicker(name='Color background colorbar', value="#16425B")
        self._widget_leg_background.param.watch(self._callback_leg_background, 'value', onlychanged=False)

    def _callback_label(self, event):
        self._label = event.new if event.new != "" else None

    def _callback_refresh(self, event):
        self.set_colorbar(0, 1, label=self._label)

    def _callback_refresh_leg(self, event):
        self.set_legend()

    def _callback_enable_colorbar(self, event):
        self.enable_colorbar = event.new
        self.set_colorbar_widget()

    def _callback_enable_legend(self, event):
        self.enable_legend = event.new
        if self.enable_legend:
            self.create_legend()
            self.sidebar.insert(1, self.legend)
        else:
            if self.legend is not None:
                self.sidebar.remove(self.legend) if self.legend in self.sidebar else None

    def set_colorbar_widget(self):
        if self.colorbar is not None:
            for pa in self.panel:
                if self.colorbar in pa:
                    pa.remove(self.colorbar)
                    break
        if self.enable_colorbar:
            if self.pos_colorbar == "horizontal":
                self.create_colorbar()
                self.colorbar.margin = (0, 0, 0, self.p_frame_left)
                self.panel[0].insert(1, self.colorbar)
            elif self.pos_colorbar == "vertical":
                self.create_colorbar()
                self.sidebar.insert(0, self.colorbar)
            self._widget_height_colorbar.value = self.col_height = self.p_frame_height if self.pos_colorbar == "vertical" \
                else round(self.p_frame_height / self._ratio)
            self._widget_width_colorbar.value = self.col_width = self.p_frame_width if self.pos_colorbar == "horizontal" \
                                           else round(self.p_frame_width / self._ratio)
            self._widget_left_colorbar.value = self.col_left = 0 if self.pos_colorbar == "vertical" else self.p_frame_left
            self._widget_top_colorbar.value = self.col_top = 0

    def _callback_colorbar_ori(self, event):
        self.pos_colorbar = event.new
        self.set_colorbar_widget()

    def _callback_top_colorbar(self, event):
        # Margins need to be tuple
        mr = list(self.colorbar.margin)
        mr[0] = event.new
        self.colorbar.margin = tuple(mr)
        self.colorbar.param.trigger('object')

    def _callback_left_colorbar(self, event):
        # Margins need to be tuple
        mr = list(self.colorbar.margin)
        mr[-1] = event.new
        self.colorbar.margin = tuple(mr)
        self.colorbar.param.trigger('object')

    def _callback_width_colorbar(self, event):
        self.colorbar.width = event.new
        self.colorbar.param.trigger('object')

    def _callback_height_colorbar(self, event):
        self.colorbar.height = event.new
        self.colorbar.param.trigger('object')

    def _callback_top_legend(self, event):
        self.leg_top = event.new
        # Margins need to be tuple
        mr = list(self.legend.margin)
        mr[0] = event.new
        self.legend.margin = tuple(mr)
        self.legend.param.trigger('object')

    def _callback_left_legend(self, event):
        self.leg_left = event.new
        mr = list(self.legend.margin)
        mr[-1] = event.new
        self.legend.margin = tuple(mr)
        self.legend.param.trigger('object')

    def _callback_width_legend(self, event):
        self.leg_width = event.new
        self.legend.width = event.new
        self.legend.param.trigger('object')

    def _callback_height_legend(self, event):
        self.leg_height = event.new
        self.legend.height = event.new
        self.legend.param.trigger('object')

    def _callback_p_frame_top(self, target, event):
        self.p_frame_top = event.new
        m = target.margin
        n = event.new
        # just changing single indices does not trigger updating of pane
        target.margin = (n, m[1], m[2], m[3])

    def _callback_p_frame_left(self, target, event):
        self.p_frame_left = event.new
        m = target.margin
        n = event.new
        target.margin = (m[0], m[1], m[2], n)

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

    def _callback_col_background(self, event):
        self.colorbar.background = event.new

    def _callback_leg_background(self, event):
        self.legend.background = event.new