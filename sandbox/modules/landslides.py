import panel as pn
import numpy
import matplotlib.pyplot as plt

from .module_main_thread import Module
from .load_save_topography import LoadSaveTopoModule


class LandslideSimulation(Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, contours=True, cmap='gist_earth_r', over='k', under='k', ** kwargs)
        self.release_area = None

        self.horizontal_flow = None
        self.vertical_flow = None

        self.flow_selector = None
        self.frame_selector = 0
        self.counter = 1
        self.simulation_frame = 0
        self.running_simulation = False

        self.npz_filename = None

        self.Load_Area = LoadSaveTopoModule(*args, **kwargs)

        self.plot_flow_frame = pn.pane.Matplotlib(plt.figure(), tight=False, height=335)
        plt.close()
        self._create_widgets()

    def setup(self):
        frame = self.sensor.get_filtered_frame()
        if self.crop:
            frame = self.crop_frame(frame)
        self.plot.render_frame(frame)
        self.projector.frame.object = self.plot.figure

    def update(self):
        # with self.lock:
        frame = self.sensor.get_filtered_frame()
        if self.crop:
            frame = self.crop_frame(frame)
        self.plot.render_frame(frame)

        self.plot_landslide_frame()

        # if aruco Module is specified: update, plot aruco markers
        if self.ARUCO_ACTIVE:
            self.update_aruco()
            self.plot.plot_aruco(self.Aruco.aruco_markers)

        self.projector.trigger()

    def plot_landslide_frame(self):
        if self.running_simulation:
            self.simulation_frame += 1
            if self.simulation_frame == (self.counter+1):
                self.simulation_frame = 0

        if self.flow_selector == 'Horizontal':
            if self.running_simulation:
                move = self.horizontal_flow[:, :, self.simulation_frame]
            else:
                move = self.horizontal_flow[:, :, self.frame_selector]

            move = numpy.round(move, decimals=1)
            move[move == 0] = numpy.nan
            move = self.Load_Area.modify_to_box_coordinates(move)
            self.plot.ax.pcolormesh(move, cmap='hot')

        elif self.flow_selector == 'Vertical':
            if self.running_simulation:
                move = self.vertical_flow[:, :, self.simulation_frame]
            else:
                move = self.vertical_flow[:, :, self.frame_selector]

            move = numpy.round(move, decimals=1)
            move[move == 0] = numpy.nan
            move = self.Load_Area.modify_to_box_coordinates(move)
            self.plot.ax.pcolormesh(move, cmap='hot')

    def plot_frame_panel(self):
        x_move = numpy.round(self.horizontal_flow[:, :, self.frame_selector], decimals=1)
        x_move[x_move == 0] = numpy.nan
        fig, (ax1, ax2) = plt.subplots(2, 1)
        hor = ax1.pcolormesh(x_move, cmap='hot')
        ax1.axis('equal')
        ax1.set_axis_off()
        ax1.set_title('Horizontal Flow')
        fig.colorbar(hor, ax=ax1)

        y_move = numpy.round(self.vertical_flow[:, :, self.frame_selector], decimals=1)
        y_move[y_move == 0] = numpy.nan
        ver = ax2.pcolormesh(y_move, cmap='hot')
        ax2.axis('equal')
        ax2.set_axis_off()
        ax2.set_title('Vertical Flow')
        fig.colorbar(ver, ax=ax2)

        self.plot_flow_frame.object = fig
        self.plot_flow_frame.param.trigger('object')

    def load_simulation_data_npz(self, infile):
        files = numpy.load(infile)
        self.vertical_flow = files['arr_0']
        self.horizontal_flow = files['arr_1']
        self.counter = self.horizontal_flow.shape[2] - 1
        print('Load successful')

    def load_release_area(self, infile):
        files = numpy.load(infile)
        self.release_area = files['arr_0']
        print('Load successful')

    # Widgets
    def show_widgets(self):
        tabs = pn.Tabs(('Controllers', self.show_tools()),
                       ('Load Simulation', self.show_load())
                       )
        return tabs

    def show_tools(self):
        widgets = pn.WidgetBox('<b>Select Flow </b>',
                               self._widget_select_direction,
                               '<b>Select Frame </b>',
                               self._widget_frame_selector,
                               '<b>Run Simulation</b>',
                               self._widget_simulation,
                               )

        rows = pn.Row(widgets, self.plot_flow_frame)
        panel = pn.Column("### Interaction widgets", rows)

        return panel

    def show_load(self):
        widgets = pn.WidgetBox('<b>Filename</b>',
                               self._widget_npz_filename,
                               '<b>Load Simulation</b>',
                               self._widget_load
                               )

        panel = pn.Column("### Load widget", widgets)

        return panel

    def _create_widgets(self):
        self._widget_frame_selector = pn.widgets.IntSlider(
            name='Frame',
            value=self.frame_selector,
            start=0,
            end=self.counter
        )
        self._widget_frame_selector.param.watch(self._callback_select_frame, 'value', onlychanged=False)

        self._widget_select_direction = pn.widgets.RadioButtonGroup(
            name='Flow direction selector',
            options=['None', 'Horizontal', 'Vertical'],
            value=['None'],
            button_type='success'
        )
        self._widget_select_direction.param.watch(self._callback_set_direction, 'value', onlychanged=False)

        self._widget_simulation = pn.widgets.RadioButtonGroup(
            name='Run or stop simulation',
            options=['Run', 'Stop'],
            value=['Stop'],
            button_type='success'
        )
        self._widget_simulation.param.watch(self._callback_simulation, 'value', onlychanged=False)

        # Load widgets
        self._widget_npz_filename = pn.widgets.TextInput(name='Choose a filename to load the simulation:')
        self._widget_npz_filename.param.watch(self._callback_filename, 'value', onlychanged=False)
        self._widget_npz_filename.value = 'simulation_data/simulation_results_for_sandbox.npz'

        self._widget_load = pn.widgets.Button(name='Load')
        self._widget_load.param.watch(self._callback_load, 'clicks', onlychanged=False)

        return True

    def _callback_set_direction(self, event):
        self.flow_selector = event.new
        self.plot_landslide_frame()

    def _callback_filename(self, event):
        self.npz_filename = event.new

    def _callback_load(self, event):
        if self.npz_filename is not None:
            self.load_simulation_data_npz(infile=self.npz_filename)
            self._widget_frame_selector.end = self.counter + 1
            self.plot_frame_panel()

    def _callback_select_frame(self, event):
        self.pause()
        self.frame_selector = event.new
        self.plot_landslide_frame()
        self.plot_frame_panel()
        self.resume()

    def _callback_simulation(self, event):
        if event.new == 'Run':
            self.running_simulation = True
        else:
            self.running_simulation = False
        self.plot_landslide_frame()
