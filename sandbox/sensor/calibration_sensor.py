import threading
import panel as pn
pn.extension()
from sandbox.sensor import Sensor
from sandbox.projector import Projector
from sandbox import _calibration_dir

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class CalibSensor: #TODO: include automatic
    """Module to calibrate the sensor"""
    def __init__(self,  calibprojector: str = None, name: str = 'kinectv2', **kwargs):
        # color map setup
        self.c_under = '#DBD053'
        self.c_over = '#DB3A34'
        # margin patches setup
        self.c_margin = '#084C61'
        self.margin_alpha = 0.5
        self.calibprojector = calibprojector
        self.sensor = Sensor(name=name, invert=False, **kwargs)
        self.projector = Projector(calibprojector=self.calibprojector, **kwargs)
        self.cmap = plt.cm.get_cmap('Greys_r')
        self.cmap.set_over(self.c_over)
        self.cmap.set_under(self.c_under)
        self.cmap.set_bad('k')

        self._refresh_panel_frame()

        #fig = plt.figure()
        self.figure = Figure()
        self.ax_notebook_frame = plt.Axes(self.figure, [0., 0., 1., 1.])
        self.figure.add_axes(self.ax_notebook_frame)

        self.calib_notebook_frame = pn.pane.Matplotlib(self.figure, tight=False, height=300)
        plt.close()  # close figure to prevent inline display

        self.projector.panel.add_periodic_callback(self.update_panel_frame, 5)

        self.frame_raw = self.sensor.get_raw_frame()
        self.ax_notebook_frame.imshow(self.frame_raw, vmin=self.sensor.s_min, vmax=self.sensor.s_max, cmap=self.cmap,
                                      origin="lower left", aspect="auto")
        self.calib_notebook_frame.param.trigger('object')
        self._create_widgets()

        #self._lock = threading.Lock()
        #self._thread = None
        #self._thread_status = 'stopped'
        #self.run()

    def _refresh_panel_frame(self):
        self.projector.ax.cla()
        self.fig_frame = self.projector.ax.imshow(self.sensor.get_frame(),
                                                      vmin=self.sensor.s_min,
                                                      vmax=self.sensor.s_max,
                                                      cmap=self.cmap,
                                                  origin="lower left",
                                                  aspect="auto")

    def update(self):
        self.update_panel_frame(self.projector.ax)
        self.update_notebook_frame(self.ax_panel_frame)

    def update_panel_frame(self):
        frame = self.sensor.get_frame()
        self.projector.ax.set_xlim(0, self.sensor.s_frame_width)
        self.projector.ax.set_ylim(0, self.sensor.s_frame_height)
        self.fig_frame.set_data(frame)

    def update_notebook_frame(self):
        """ Adds margin patches to the current plot object.
        This is only useful when an uncropped dataframe is passed.
        """

        self.ax_notebook_frame.cla()
        self.ax_notebook_frame.imshow(self.frame_raw, vmin=self.sensor.s_min, vmax=self.sensor.s_max, cmap=self.cmap,
                                      origin="lower left", aspect="auto")

        rec_t = plt.Rectangle((0, self.sensor.s_height - self.sensor.s_top), self.sensor.s_width, self.sensor.s_top,
                              fc=self.c_margin, alpha=self.margin_alpha)
        rec_r = plt.Rectangle((self.sensor.s_width - self.sensor.s_right, 0), self.sensor.s_right, self.sensor.s_height,
                              fc=self.c_margin, alpha=self.margin_alpha)
        rec_b = plt.Rectangle((0, 0), self.sensor.s_width, self.sensor.s_bottom,
                              fc=self.c_margin, alpha=self.margin_alpha)
        rec_l = plt.Rectangle((0, 0), self.sensor.s_left, self.sensor.s_height,
                              fc=self.c_margin, alpha=self.margin_alpha)
        self.ax_notebook_frame.add_patch(rec_t)
        self.ax_notebook_frame.add_patch(rec_r)
        self.ax_notebook_frame.add_patch(rec_b)
        self.ax_notebook_frame.add_patch(rec_l)
        self.calib_notebook_frame.param.trigger('object')

    def calibrate_sensor(self):
        widgets = pn.WidgetBox('<b>Load a projector calibration file</b>',
                               self._widget_json_filename_load_projector,
                               self._widget_json_load_projector,
                               '<b>Distance from edges (pixel)</b>',
                               self._widget_s_top,
                               self._widget_s_right,
                               self._widget_s_bottom,
                               self._widget_s_left,
                               #self._widget_s_enable_auto_cropping,
                               #self._widget_s_automatic_cropping,
                               pn.layout.VSpacer(height=5),
                               '<b>Distance from sensor (mm)</b>',
                               self._widget_s_min,
                               self._widget_s_max,
                               self._widget_refresh_frame)
        box = pn.Column('<b>Physical dimensions of the sandbox</b>',
                        self._widget_box_width,
                        self._widget_box_height,
                        )
        save = pn.Column('<b>Save file</b>',
                         self._widget_json_filename,
                         self._widget_json_save
                         )

        rows = pn.Row(widgets, self.calib_notebook_frame)
        panel = pn.Column('## Sensor calibration', rows)
        tabs = pn.Tabs(('Calibration', panel),
                       ("Box dimensions", box),
                       ("Save files", save)
                       )
        return tabs

    def _create_widgets(self):
        # sensor widgets and links

        self._widget_s_top = pn.widgets.IntSlider(name='Sensor top margin',
                                                  bar_color=self.c_margin,
                                                  value=self.sensor.s_top,
                                                  start=1,
                                                  end=self.sensor.s_height)
        self._widget_s_top.param.watch(self._callback_s_top, 'value', onlychanged=False)

        self._widget_s_right = pn.widgets.IntSlider(name='Sensor right margin',
                                                    bar_color=self.c_margin,
                                                    value=self.sensor.s_right,
                                                    start=1,
                                                    end=self.sensor.s_width)
        self._widget_s_right.param.watch(self._callback_s_right, 'value', onlychanged=False)

        self._widget_s_bottom = pn.widgets.IntSlider(name='Sensor bottom margin',
                                                     bar_color=self.c_margin,
                                                     value=self.sensor.s_bottom,
                                                     start=1,
                                                     end=self.sensor.s_height)
        self._widget_s_bottom.param.watch(self._callback_s_bottom, 'value', onlychanged=False)

        self._widget_s_left = pn.widgets.IntSlider(name='Sensor left margin',
                                                   bar_color=self.c_margin,
                                                   value=self.sensor.s_left,
                                                   start=1,
                                                   end=self.sensor.s_width)
        self._widget_s_left.param.watch(self._callback_s_left, 'value', onlychanged=False)

        self._widget_s_min = pn.widgets.IntSlider(name='Vertical minimum',
                                                  bar_color=self.c_under,
                                                  value=self.sensor.s_min,
                                                  start=0,
                                                  end=2000)
        self._widget_s_min.param.watch(self._callback_s_min, 'value', onlychanged=False)

        self._widget_s_max = pn.widgets.IntSlider(name='Vertical maximum',
                                                  bar_color=self.c_over,
                                                  value=self.sensor.s_max,
                                                  start=0,
                                                  end=2000)
        self._widget_s_max.param.watch(self._callback_s_max, 'value', onlychanged=False)

        # Auto cropping widgets:

        #self._widget_s_enable_auto_cropping = pn.widgets.Checkbox(name='Enable Automatic Cropping', value=False)
        #self._widget_s_enable_auto_cropping.param.watch(self._callback_enable_auto_cropping, 'value',
        #                                                onlychanged=False)

        #self._widget_s_automatic_cropping = pn.widgets.Button(name="Crop", button_type="success")
        #self._widget_s_automatic_cropping.param.watch(self._callback_automatic_cropping, 'clicks',
        #                                              onlychanged=False)

        # box widgets:

        # self._widget_s_enable_auto_calibration = CheckboxGroup(labels=["Enable Automatic Sensor Calibration"],
        #                                                                  active=[1])
        self._widget_box_width = pn.widgets.IntSlider(name='width of sandbox in mm',
                                                      bar_color=self.c_margin,
                                                      value=int(self.sensor.box_width),
                                                      start=1,
                                                      end=2000)
        self._widget_box_width.param.watch(self._callback_box_width, 'value', onlychanged=False)

        # self._widget_s_automatic_calibration = pn.widgets.Toggle(name="Run", button_type="success")
        self._widget_box_height = pn.widgets.IntSlider(name='height of sandbox in mm',
                                                       bar_color=self.c_margin,
                                                       value=int(self.sensor.box_height),
                                                       start=1,
                                                       end=2000)
        self._widget_box_height.param.watch(self._callback_box_height, 'value', onlychanged=False)

        # refresh button

        self._widget_refresh_frame = pn.widgets.Button(name='Refresh sensor frame\n(3 sec. delay)!')
        self._widget_refresh_frame.param.watch(self._callback_refresh_frame, 'clicks', onlychanged=False)

        # save selection

        # Only for reading files --> Is there no location picker in panel widgets???
        # self._widget_json_location = pn.widgets.FileInput(name='JSON location')
        self._widget_json_filename = pn.widgets.TextInput(name='Choose a calibration filename:')
        self._widget_json_filename.param.watch(self._callback_json_filename, 'value', onlychanged=False)
        self._widget_json_filename.value = _calibration_dir + 'my_sensor_calibration.json'

        self._widget_json_save = pn.widgets.Button(name='Save calibration')
        self._widget_json_save.param.watch(self._callback_json_save, 'clicks', onlychanged=False)

        self._widget_json_filename_load_projector = pn.widgets.TextInput(name='Choose the projector calibration filename:')
        self._widget_json_filename_load_projector.param.watch(self._callback_json_filename_load_projector, 'value', onlychanged=False)
        self._widget_json_filename_load_projector.value = _calibration_dir + 'my_projector_calibration.json'

        self._widget_json_load_projector = pn.widgets.Button(name='Load calibration')
        self._widget_json_load_projector.param.watch(self._callback_json_load_projector, 'clicks', onlychanged=False)

        return True

        # sensor callbacks
    def _callback_s_top(self, event):
        self.sensor.s_top = event.new
        # change plot and trigger panel update
        self.update_notebook_frame()

    def _callback_s_right(self, event):
        self.sensor.s_right = event.new
        #self._refresh_panel_frame() #TODO: dirty workaround
        self.update_notebook_frame()

    def _callback_s_bottom(self, event):
        self.sensor.s_bottom = event.new
        self.update_notebook_frame()

    def _callback_s_left(self, event):
        self.sensor.s_left = event.new
        #self._refresh_panel_frame()  # TODO: dirty workaround
        self.update_notebook_frame()

    def _callback_s_min(self, event):
        self.sensor.s_min = event.new
        #self._refresh_panel_frame()  # TODO: dirty workaround
        self.update_notebook_frame()

    def _callback_s_max(self, event):
        self.sensor.s_max = event.new
        #self._refresh_panel_frame()  # TODO: dirty workaround
        self.update_notebook_frame()

    def _callback_refresh_frame(self, event):
        plt.pause(3)
        # only here, get a new frame before updating the plot
        self.frame_raw = self.sensor.get_raw_frame()
        self.update_notebook_frame()

    def _callback_json_filename(self, event):
        self.sensor.json_filename = event.new

    def _callback_json_save(self, event):
        if self.sensor.json_filename is not None:
            self.sensor.save_json(file=self.sensor.json_filename)

    def _callback_json_filename_load_projector(self, event):
        self.calibprojector = event.new

    def _callback_json_load_projector(self, event):
        if self.calibprojector is not None:
            self.projector = Projector(self.calibprojector)

    def _callback_box_width(self, event):
        self.sensor.box_width = float(event.new)

    def _callback_box_height(self, event):
        self.sensor.box_height = float(event.new)

    """def _callback_enable_auto_calibration(self, event):
        self.automatic_calibration = event.new
        if self.automatic_calibration == True:
            self.plot.render_frame(self.Aruco.p_arucoMarker(), vmin=0, vmax=256)
            self.projector.frame.object = self.plot.figure
        else:
            self.plot.create_empty_frame()
            self.projector.frame.object = self.plot.figure

    def _callback_automatic_calibration(self, event):
        if self.automatic_calibration == True:
            p_frame_left, p_frame_top, p_frame_width, p_frame_height = self.Aruco.move_image()
            self.calib.p_frame_left = p_frame_left
            self.calib.p_frame_top = p_frame_top
            self._widget_p_frame_left.value = self.calib.p_frame_left
            self._widget_p_frame_top.value = self.calib.p_frame_top
            self.calib.p_frame_width = p_frame_width
            self.calib.p_frame_height = p_frame_height
            self._widget_p_frame_width.value = self.calib.p_frame_width
            self._widget_p_frame_height.value = self.calib.p_frame_height
            self.plot.render_frame(self.Aruco.p_arucoMarker(), vmin=0, vmax=256)
            self.projector.frame.object = self.plot.figure
            self.update_calib_plot()


    def _callback_enable_auto_cropping(self, event):
        self.automatic_cropping = event.new


    def _callback_automatic_cropping(self, event):
        if self.automatic_cropping == True:
            self.pause()
            s_top, s_left, s_bottom, s_right = self.Aruco.crop_image_aruco()
            self.calib.s_top = s_top
            self.calib.s_bottom = s_bottom
            self.calib.s_left = s_left
            self.calib.s_right = s_right
            self._widget_s_top.value = self.calib.s_top
            self._widget_s_bottom.value = self.calib.s_bottom
            self._widget_s_left.value = self.calib.s_left
            self._widget_s_right.value = self.calib.s_right
            self.update_calib_plot()
            self.resume()"""