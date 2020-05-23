import panel as pn
import matplotlib.pyplot as plt
from time import sleep

from sandbox.plot.plot import Plot
from sandbox.modules.module_main_thread import Module

class CalibModule(Module):
    """
    Module for calibration and responsive visualization
    """

    def __init__(self, *args, **kwargs):
        # customization
        self.c_under = '#DBD053'
        self.c_over = '#DB3A34'
        self.c_margin = '#084C61'
        self.automatic_calibration = False
        self.automatic_cropping = False
        # call parents' class init, use greyscale colormap as standard and extreme color labeling
        super().__init__(*args, contours=True, over=self.c_over, cmap='Greys_r', under=self.c_under, **kwargs)

        self.json_filename = None

        # sensor calibration visualization
        pn.extension()
        self.calib_frame = None  # snapshot of sensor frame, only updated with refresh button
        self.calib_plot = Plot(self.calib, margins=True, contours=True,
                               margin_color=self.c_margin,
                               cmap='Greys_r', over=self.c_over, under=self.c_under)#, **kwargs)
        self.calib_panel_frame = pn.pane.Matplotlib(plt.figure(), tight=False, height=335)
        plt.close()  # close figure to prevent inline display
        self._create_widgets()

    # standard methods
    def setup(self):
        frame = self.sensor.get_filtered_frame()
        if self.crop:
            frame = self.crop_frame(frame)
        self.plot.render_frame(frame)
        self.projector.frame.object = self.plot.figure

        # sensor calibration visualization
        self.calib_frame = self.sensor.get_filtered_frame()
        self.calib_plot.render_frame(self.calib_frame)
        self.calib_panel_frame.object = self.calib_plot.figure

    def update(self):
        frame = self.sensor.get_filtered_frame()
        if self.crop:
            frame = self.crop_frame(frame)
        self.plot.render_frame(frame, vmin=self.calib.s_min, vmax=self.calib.s_max)

        # if aruco Module is specified:search, update, plot aruco markers
        if self.ARUCO_ACTIVE:
            self.update_aruco()
            self.plot.plot_aruco(self.Aruco.aruco_markers)

        self.projector.trigger()

    def update_calib_plot(self):
        self.calib_plot.render_frame(self.calib_frame)
        self.calib_panel_frame.param.trigger('object')

    # layouts
    def calibrate_projector(self):
        widgets = pn.WidgetBox(self._widget_p_frame_top,
                               self._widget_p_frame_left,
                               self._widget_p_frame_width,
                               self._widget_p_frame_height,
                               self._widget_p_enable_auto_calibration,
                               self._widget_p_automatic_calibration)
        panel = pn.Column("### Projector dashboard arrangement", widgets)
        return panel

    def calibrate_sensor(self):
        widgets = pn.WidgetBox('<b>Distance from edges (pixel)</b>',
                               self._widget_s_top,
                               self._widget_s_right,
                               self._widget_s_bottom,
                               self._widget_s_left,
                               self._widget_s_enable_auto_cropping,
                               self._widget_s_automatic_cropping,
                               pn.layout.VSpacer(height=5),
                               '<b>Distance from sensor (mm)</b>',
                               self._widget_s_min,
                               self._widget_s_max,
                               self._widget_refresh_frame
                               )
        rows = pn.Row(widgets, self.calib_panel_frame)
        panel = pn.Column('### Sensor calibration', rows)
        return panel

    def calibrate_box(self):
        widgets = pn.WidgetBox('<b>Physical dimensions of the sandbox)</b>',
                               self._widget_box_width,
                               self._widget_box_height,
                               )
        panel = pn.Column('### box calibration', widgets)
        return panel

    def show_widgets(self):
        tabs = pn.Tabs(('Projector', self.calibrate_projector()),
                       ('Sensor', self.calibrate_sensor()),
                       ('Box Dimensions', self.calibrate_box()),
                       ('Save', pn.WidgetBox(self._widget_json_filename,
                                             self._widget_json_save))
                       )
        return tabs

    def _create_widgets(self):

        # projector widgets and links

        self._widget_p_frame_top = pn.widgets.IntSlider(name='Main frame top margin',
                                                        value=self.calib.p_frame_top,
                                                        start=0,
                                                        end=self.calib.p_height - 20)
        self._widget_p_frame_top.link(self.projector.frame, callbacks={'value': self._callback_p_frame_top})

        self._widget_p_frame_left = pn.widgets.IntSlider(name='Main frame left margin',
                                                         value=self.calib.p_frame_left,
                                                         start=0,
                                                         end=self.calib.p_width - 20)
        self._widget_p_frame_left.link(self.projector.frame, callbacks={'value': self._callback_p_frame_left})

        self._widget_p_frame_width = pn.widgets.IntSlider(name='Main frame width',
                                                          value=self.calib.p_frame_width,
                                                          start=10,
                                                          end=self.calib.p_width)
        self._widget_p_frame_width.link(self.projector.frame, callbacks={'value': self._callback_p_frame_width})

        self._widget_p_frame_height = pn.widgets.IntSlider(name='Main frame height',
                                                           value=self.calib.p_frame_height,
                                                           start=10,
                                                           end=self.calib.p_height)
        self._widget_p_frame_height.link(self.projector.frame, callbacks={'value': self._callback_p_frame_height})

        # Auto- Calibration widgets

        self._widget_p_enable_auto_calibration = pn.widgets.Checkbox(name='Enable Automatic Calibration', value=False)
        self._widget_p_enable_auto_calibration.param.watch(self._callback_enable_auto_calibration, 'value',
                                                           onlychanged=False)

        self._widget_p_automatic_calibration = pn.widgets.Button(name="Run", button_type="success")
        self._widget_p_automatic_calibration.param.watch(self._callback_automatic_calibration, 'clicks',
                                                         onlychanged=False)

        # sensor widgets and links

        self._widget_s_top = pn.widgets.IntSlider(name='Sensor top margin',
                                                  bar_color=self.c_margin,
                                                  value=self.calib.s_top,
                                                  start=1,
                                                  end=self.calib.s_height)
        self._widget_s_top.param.watch(self._callback_s_top, 'value', onlychanged=False)

        self._widget_s_right = pn.widgets.IntSlider(name='Sensor right margin',
                                                    bar_color=self.c_margin,
                                                    value=self.calib.s_right,
                                                    start=1,
                                                    end=self.calib.s_width)
        self._widget_s_right.param.watch(self._callback_s_right, 'value', onlychanged=False)

        self._widget_s_bottom = pn.widgets.IntSlider(name='Sensor bottom margin',
                                                     bar_color=self.c_margin,
                                                     value=self.calib.s_bottom,
                                                     start=1,
                                                     end=self.calib.s_height)
        self._widget_s_bottom.param.watch(self._callback_s_bottom, 'value', onlychanged=False)

        self._widget_s_left = pn.widgets.IntSlider(name='Sensor left margin',
                                                   bar_color=self.c_margin,
                                                   value=self.calib.s_left,
                                                   start=1,
                                                   end=self.calib.s_width)
        self._widget_s_left.param.watch(self._callback_s_left, 'value', onlychanged=False)

        self._widget_s_min = pn.widgets.IntSlider(name='Vertical minimum',
                                                  bar_color=self.c_under,
                                                  value=self.calib.s_min,
                                                  start=0,
                                                  end=2000)
        self._widget_s_min.param.watch(self._callback_s_min, 'value', onlychanged=False)

        self._widget_s_max = pn.widgets.IntSlider(name='Vertical maximum',
                                                  bar_color=self.c_over,
                                                  value=self.calib.s_max,
                                                  start=0,
                                                  end=2000)
        self._widget_s_max.param.watch(self._callback_s_max, 'value', onlychanged=False)

        # Auto cropping widgets:

        self._widget_s_enable_auto_cropping = pn.widgets.Checkbox(name='Enable Automatic Cropping', value=False)
        self._widget_s_enable_auto_cropping.param.watch(self._callback_enable_auto_cropping, 'value',
                                                        onlychanged=False)

        self._widget_s_automatic_cropping = pn.widgets.Button(name="Crop", button_type="success")
        self._widget_s_automatic_cropping.param.watch(self._callback_automatic_cropping, 'clicks',
                                                      onlychanged=False)

        # box widgets:

        # self._widget_s_enable_auto_calibration = CheckboxGroup(labels=["Enable Automatic Sensor Calibration"],
        #                                                                  active=[1])
        self._widget_box_width = pn.widgets.IntSlider(name='width of sandbox in mm',
                                                      bar_color=self.c_margin,
                                                      value=int(self.calib.box_width),
                                                      start=1,
                                                      end=2000)
        self._widget_box_width.param.watch(self._callback_box_width, 'value', onlychanged=False)

        # self._widget_s_automatic_calibration = pn.widgets.Toggle(name="Run", button_type="success")
        self._widget_box_height = pn.widgets.IntSlider(name='height of sandbox in mm',
                                                       bar_color=self.c_margin,
                                                       value=int(self.calib.box_height),
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
        self._widget_json_filename.value = '../../calibration_files/my_calibration.json'

        self._widget_json_save = pn.widgets.Button(name='Save calibration')
        self._widget_json_save.param.watch(self._callback_json_save, 'clicks', onlychanged=False)

        return True

    # projector callbacks

    def _callback_p_frame_top(self, target, event):
        self.pause()
        # set value in calib
        self.calib.p_frame_top = event.new
        m = target.margin
        n = event.new
        # just changing single indices does not trigger updating of pane
        target.margin = [n, m[1], m[2], m[3]]
        self.resume()

    def _callback_p_frame_left(self, target, event):
        self.pause()
        self.calib.p_frame_left = event.new
        m = target.margin
        n = event.new
        target.margin = [m[0], m[1], m[2], n]
        self.resume()

    def _callback_p_frame_width(self, target, event):
        self.pause()
        self.calib.p_frame_width = event.new
        target.width = event.new
        target.param.trigger('object')
        self.resume()

    def _callback_p_frame_height(self, target, event):
        self.pause()
        self.calib.p_frame_height = event.new
        target.height = event.new
        target.param.trigger('object')
        self.resume()

    # sensor callbacks

    def _callback_s_top(self, event):
        self.pause()
        # set value in calib
        self.calib.s_top = event.new
        # change plot and trigger panel update
        self.update_calib_plot()
        self.resume()

    def _callback_s_right(self, event):
        self.pause()
        self.calib.s_right = event.new
        self.update_calib_plot()
        self.resume()

    def _callback_s_bottom(self, event):
        self.pause()
        self.calib.s_bottom = event.new
        self.update_calib_plot()
        self.resume()

    def _callback_s_left(self, event):
        self.pause()
        self.calib.s_left = event.new
        self.update_calib_plot()
        self.resume()

    def _callback_s_min(self, event):
        self.pause()
        self.calib.s_min = event.new
        self.update_calib_plot()
        self.resume()

    def _callback_s_max(self, event):
        self.pause()
        self.calib.s_max = event.new
        self.update_calib_plot()
        self.resume()

    def _callback_refresh_frame(self, event):
        self.pause()
        sleep(3)
        # only here, get a new frame before updating the plot
        self.calib_frame = self.sensor.get_filtered_frame()
        self.update_calib_plot()
        self.resume()

    def _callback_json_filename(self, event):
        self.json_filename = event.new

    def _callback_json_save(self, event):
        if self.json_filename is not None:
            self.calib.save_json(file=self.json_filename)

    ### box dimensions callbacks:

    def _callback_box_width(self, event):
        self.pause()
        self.calib.box_width = float(event.new)
        # self.update_calib_plot()
        self.resume()

    def _callback_box_height(self, event):
        self.pause()
        self.calib.box_height = float(event.new)
        # self.update_calib_plot()
        self.resume()

    ### Automatic Calibration callback

    def _callback_enable_auto_calibration(self, event):
        self.automatic_calibration = event.new
        if self.automatic_calibration == True:
            self.plot.render_frame(self.Aruco.p_arucoMarker(),  vmin=0, vmax=256)
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
            self._widget_s_top.value=self.calib.s_top
            self._widget_s_bottom.value = self.calib.s_bottom
            self._widget_s_left.value = self.calib.s_left
            self._widget_s_right.value = self.calib.s_right
            self.update_calib_plot()
            self.resume()
