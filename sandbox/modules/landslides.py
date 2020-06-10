import sys, os
import panel as pn
import numpy
import matplotlib.pyplot as plt
import matplotlib

from .module_main_thread import Module
from .load_save_topography import LoadSaveTopoModule


class LandslideSimulation(Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, contours=True, cmap='gist_earth_r', over='k', under='k', ** kwargs)
        self.release_area = None
        self.release_area_all = None

        self.horizontal_flow = None
        self.vertical_flow = None

        self.flow_selector = None
        self.frame_selector = 0
        self.counter = 1
        self.simulation_frame = 0
        self.running_simulation = False

        self.simulation_folder = "C:/Users/Admin/PycharmProjects/open_AR_Sandbox/notebooks/tutorials/07_LandslideSimulation/simulation_data/"
        self.release_folder = "C:/Users/Admin/PycharmProjects/open_AR_Sandbox/notebooks/tutorials/07_LandslideSimulation/saved_ReleaseAreas/"
        self.release_options = None
        self.release_id = None
        self.release_id_all = ['None']
        self.box_release_area = False

        self.Load_Area = LoadSaveTopoModule(*args, **kwargs)

        self.plot_flow_frame = pn.pane.Matplotlib(plt.figure(), tight=False, height=335)
        plt.close()

    def setup(self):
        frame = self.sensor.get_frame()
        if self.crop:
            frame = self.crop_frame(frame)
        self.plot.render_frame(frame)
        self.projector.frame.object = self.plot.figure
        self._create_widgets()

    def update(self):
        # with self.lock:
        frame = self.sensor.get_frame()
        if self.crop:
            frame = self.crop_frame(frame)
        self.plot.render_frame(frame)

        if self.box_release_area:
            self.show_box_release(self.release_area)

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

        if self.flow_selector == 'Height':
            if self.running_simulation:
                move = self.horizontal_flow[:, :, self.simulation_frame]
            else:
                move = self.horizontal_flow[:, :, self.frame_selector]

            move = numpy.round(move, decimals=1)
            move[move == 0] = numpy.nan
            move = self.Load_Area.modify_to_box_coordinates(move)
            self.plot.ax.pcolormesh(move, cmap='hot')

        elif self.flow_selector == 'Velocity':
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
        ax1.set_title('Flow Height')
        fig.colorbar(hor, ax=ax1, label='meter')

        y_move = numpy.round(self.vertical_flow[:, :, self.frame_selector], decimals=1)
        y_move[y_move == 0] = numpy.nan
        ver = ax2.pcolormesh(y_move, cmap='hot')
        ax2.axis('equal')
        ax2.set_axis_off()
        ax2.set_title('Flow Velocity')
        fig.colorbar(ver, ax=ax2, label='meter/sec')

        self.plot_flow_frame.object = fig
        self.plot_flow_frame.param.trigger('object')
        plt.close()

    def load_simulation_data_npz(self, infile):
        files = numpy.load(infile)
        self.vertical_flow = files['arr_0']
        self.horizontal_flow = files['arr_1']
        self.counter = self.horizontal_flow.shape[2] - 1
        print('Load successful')

    def load_release_area(self, data_path):
        self.release_options = []
        self.release_id_all = []
        self.release_area_all = []
        list_files = os.listdir(data_path)
        for i in list_files:
            temp = [str(s) for s in i if s.isdigit()]
            if len(temp) > 0:
                if temp[0] == self.Load_Area.file_id:
                    self.release_options.append(i)
                    self.release_id_all.append(temp[-1])
                    self.release_area_all.append(numpy.load(data_path+i))

    def show_box_release(self, xy):
        patch = matplotlib.patches.Polygon(xy, fill=False, edgecolor='white')
        self.plot.ax.add_patch(patch)

    def select_simulation_data(self, simulation_path, release_id):
        list_files = os.listdir(simulation_path)
        for i in list_files:
            temp = [str(s) for s in i if s.isdigit()]
            if len(temp) > 0:
                if temp[0] == self.Load_Area.file_id and temp[2] == release_id:
                    file_name = i
        file_location = simulation_path+file_name
        self.load_simulation_data_npz(file_location)

    def modify_to_box_coordinates(self, id):
        temp = self.release_area_all[int(id) - 1]
        self.release_area = numpy.vstack((temp[:,0]+self.Load_Area.box_origin[0], temp[:,1]+self.Load_Area.box_origin[1])).T

    # Widgets
    def show_widgets(self):
        self._create_widgets()
        tabs = pn.Tabs(('Load and Save Module', self.show_loadsave_widgets()),
                       ('Landslide simulation module', self.show_landslide_widgets()))
        return tabs

    def show_loadsave_widgets(self):
        self.Load_Area._create_widgets()
        tabs = pn.Tabs(('Activate Module', pn.WidgetBox(self._widget_activate_LoadSaveModule)),
                       ('Box widgets', self.Load_Area.widgets_box()),
                       ('Release area widgets', self.Load_Area.widgets_release_area()),
                       ('Load Topography', self.Load_Area.widgets_load()),
                       ('Save Topography', self.Load_Area.widgets_save()),
                       ('Plot', self.Load_Area.widget_plot_module())
                       )
        return tabs

    def show_landslide_widgets(self):
        self._create_widgets()
        tabs = pn.Tabs(("Activate Module", pn.WidgetBox(self._widget_activate_LandslideModule)),
                       ('Controllers', self.widgets_controller_simulation()),
                       ('Load Simulation', self.widgets_load_simulation()),
                       ("Plot", self.widget_plot_module())
                       )
        return tabs

    def widgets_controller_simulation(self):
        widgets = pn.Column("### Interaction widgets",
                            self._widget_show_release,
                            '<b>Select Flow </b>',
                            self._widget_select_direction,
                            '<b>Select Frame </b>',
                            self._widget_frame_selector,
                            '<b>Run Simulation</b>',
                            self._widget_simulation
                            )
        panel = pn.Row(widgets, self.plot_flow_frame)
        return panel

    def widgets_load_simulation(self):
        col1 = pn.Column("## Load widget",
                            '<b>File path</b>',
                            self._widget_simulation_folder,
                            '<b> Load possible release areas</b>',
                            self._widget_load)
        col2 = pn.Column('<b>Load Simulation</b>',
                          self._widget_load_release_area,
                          '<b>Select a release area</b>',
                          pn.WidgetBox(self._widget_available_release_areas))
        panel = pn.Row(col1, col2)
        return panel

    def _create_widgets(self):
        self._widget_activate_LoadSaveModule = pn.widgets.Button(name='Activate Load Save Module', button_type="success")
        self._widget_activate_LoadSaveModule.param.watch(self._callback_activate_LoadSaveModule, 'clicks', onlychanged=False)

        self._widget_activate_LandslideModule = pn.widgets.Button(name='Activate Landslide Module',
                                                                 button_type="success")
        self._widget_activate_LandslideModule.param.watch(self._callback_activate_LandslideModule, 'clicks',
                                                         onlychanged=False)

        self._widget_frame_selector = pn.widgets.IntSlider(
            name='5 seconds time step',
            value=self.frame_selector,
            start=0,
            end=self.counter
        )
        self._widget_frame_selector.param.watch(self._callback_select_frame, 'value', onlychanged=False)

        self._widget_select_direction = pn.widgets.RadioButtonGroup(
            name='Flow history selector',
            options=['None', 'Height', 'Velocity'],
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
        self._widget_simulation_folder = pn.widgets.TextInput(name='Specify the folder path to load the simulation:')
        self._widget_simulation_folder.param.watch(self._callback_filename, 'value', onlychanged=False)
        self._widget_simulation_folder.value = self.simulation_folder

        self._widget_load = pn.widgets.Button(name='Refresh list', button_type='warning')
        self._widget_load.param.watch(self._callback_load_files_folder, 'clicks', onlychanged=False)

        self._widget_available_release_areas = pn.widgets.RadioBoxGroup(name='Available release areas',
                                                                     options=self.release_id_all,
                                                                     inline=False,
                                                                        value=None)
        self._widget_available_release_areas.param.watch(self._callback_available_release_areas, 'value',
                                                      onlychanged=False)

        self._widget_load_release_area = pn.widgets.Button(name='Load selected release area',
                                                                 button_type="success")
        self._widget_load_release_area.param.watch(self._callback_load_release_area, 'clicks',
                                                         onlychanged=False)

        self._widget_show_release = pn.widgets.Checkbox(name='Show release area', value=self.box_release_area, disabled=True)
        self._widget_show_release.param.watch(self._callback_show_release, 'value',
                                                    onlychanged=False)

        return True

    def _callback_set_direction(self, event):
        self.flow_selector = event.new
        self.plot_landslide_frame()

    def _callback_filename(self, event):
        self.simulation_folder = event.new

    def _callback_load_files_folder(self, event):
        if self.simulation_folder is not None:
            self.load_release_area(self.release_folder)
            self._widget_available_release_areas.options = self.release_id_all
            self._widget_available_release_areas.sizing_mode = "scale_both"

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

    def _callback_activate_LoadSaveModule(self, event):
        self.stop()
        self.Load_Area.setup()
        self.Load_Area.run()

    def _callback_activate_LandslideModule(self, event):
        self.Load_Area.stop()
        self.setup()
        self.run()

    def _callback_available_release_areas(self, event):
        if event.new is not None:
            self.release_id = event.new
            self.modify_to_box_coordinates(self.release_id)
            self.box_release_area = True
            self._widget_show_release.disabled = False
            self._widget_show_release.value = self.box_release_area

    def _callback_load_release_area(self, event):
        if self.simulation_folder is not None:
            self.select_simulation_data(self.simulation_folder, self.release_id)
            self._widget_frame_selector.end = self.counter + 1
            self.plot_frame_panel()

    def _callback_show_release(self, event):
        self.box_release_area = event.new
