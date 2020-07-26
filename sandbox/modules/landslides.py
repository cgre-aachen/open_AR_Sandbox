import os
import panel as pn
pn.extension()
import numpy
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.figure import Figure
from .template import ModuleTemplate
from .load_save_topography import LoadSaveTopoModule
from sandbox import _test_data


class LandslideSimulation(ModuleTemplate):
    """
    Module to show the results of landslides simulations
    """

    def __init__(self, *args, extent: list = None, **kwargs):
        if extent is not None:
            self.vmin = extent[4]
            self.vmax = extent[5]

        self.release_area = None
        self.release_area_all = None

        self.height_flow = None
        self.velocity_flow = None

        self.flow_selector = None
        self.frame_selector = 0
        self.counter = 1
        self.simulation_frame = 0
        self.running_simulation = False

        self.simulation_folder = _test_data['landslide_simulation']
        self.release_folder = _test_data['landslide_release']
        self.topo_folder = _test_data['landslide_topo']

        self.release_options = None
        self.release_id = None
        self.release_id_all = ['None']
        self.box_release_area = False

        self.Load_Area = LoadSaveTopoModule(*args, extent=extent, **kwargs)
        self.Load_Area.data_path = self.topo_folder

        self.figure = Figure()
        self.ax1 = self.figure.add_subplot(211)
        self.ax2 = self.figure.add_subplot(212)
        self.plot_flow_frame = pn.pane.Matplotlib(self.figure, tight=False, height=500)
        plt.close(self.figure)  # close figure to prevent inline display

    def update(self, frame, ax, extent, marker=[]):
        self.Load_Area.plot(frame, ax)
        self.plot(frame, ax)
        cmap = plt.get_cmap('gist_earth_r')
        norm = None
        return frame, ax, extent, cmap, norm

    def plot(self, frame, ax):
        if self.box_release_area:
            self.show_box_release(ax, self.release_area)
        self.plot_landslide_frame(ax)

    def plot_landslide_frame(self, ax):
        """
        Plot from the current landslide frame depending of the type of flow (Height or velocity)
        Args:
            ax: The image to plot the frame
        Returns:
        """
        if self.running_simulation:
            self.simulation_frame += 1
            if self.simulation_frame == (self.counter+1):
                self.simulation_frame = 0

        if self.flow_selector == 'Height':
            if self.running_simulation:
                move = self.height_flow[:, :, self.simulation_frame]
            else:
                move = self.height_flow[:, :, self.frame_selector]

            move = numpy.round(move, decimals=1)
            move[move == 0] = numpy.nan
            move = self.Load_Area.modify_to_box_coordinates(move)
            ax.pcolormesh(move, cmap='hot', shading='gouraud')

        elif self.flow_selector == 'Velocity':
            if self.running_simulation:
                move = self.velocity_flow[:, :, self.simulation_frame]
            else:
                move = self.velocity_flow[:, :, self.frame_selector]

            move = numpy.round(move, decimals=1)
            move[move == 0] = numpy.nan
            move = self.Load_Area.modify_to_box_coordinates(move)
            ax.pcolormesh(move, cmap='hot', shading='gouraud')

    def plot_frame_panel(self):
        """Update the current frame to be displayed in the panel server"""
        #self.figure.clf()
        self.ax1.cla()
        self.ax2.cla()
        #self.ax1.images[-1].colorbar.remove()
        #self.ax2.images[-1].colorbar.remove()


        x_move = numpy.round(self.height_flow[:, :, self.frame_selector], decimals=1)
        x_move[x_move == 0] = numpy.nan
        hor = self.ax1.pcolormesh(x_move, cmap='hot')
        self.ax1.axis('equal')
        self.ax1.set_axis_off()
        self.ax1.set_title('Flow Height')
        self.figure.colorbar(hor, ax=self.ax1, label='meter')

        y_move = numpy.round(self.velocity_flow[:, :, self.frame_selector], decimals=1)
        y_move[y_move == 0] = numpy.nan
        ver = self.ax2.pcolormesh(y_move, cmap='hot')
        self.ax2.axis('equal')
        self.ax2.set_axis_off()
        self.ax2.set_title('Flow Velocity')
        self.figure.colorbar(ver, ax=self.ax2, label='meter/sec')

        self.plot_flow_frame.param.trigger('object')

    def load_simulation_data_npz(self, infile):
        """Load landslide simulation from a .npz file """
        files = numpy.load(infile)
        self.velocity_flow = files['arr_0']
        self.height_flow = files['arr_1']
        self.counter = self.height_flow.shape[2] - 1
        print('Load successful')

    def load_release_area(self, data_path):
        """load all possible release areas from the same topography to show the simulation"""
        self.release_options = []
        self.release_id_all = []
        self.release_area_all = []
        list_files = os.listdir(data_path)
        for i in list_files:
            temp = [str(s) for s in i if s.isdigit()]
            if len(temp) > 0:
                if temp[-2] == self.Load_Area.file_id:
                    self.release_options.append(i)
                    self.release_id_all.append(temp[-1])
                    self.release_area_all.append(numpy.load(data_path+i))

    def show_box_release(self, ax, xy):
        """Show the options for painting release areas at origin xy """
        patch = matplotlib.patches.Polygon(xy, fill=False, edgecolor='white')
        ax.add_patch(patch)

    def select_simulation_data(self, simulation_path, release_id):
        """Make sure that the simulation data corresponds to the loaded topography"""
        list_files = os.listdir(simulation_path)
        for i in list_files:
            temp = [str(s) for s in i if s.isdigit()]
            if len(temp) > 0:
                if temp[0] == self.Load_Area.file_id and temp[2] == release_id:
                    file_name = i
        file_location = simulation_path+file_name
        self.load_simulation_data_npz(file_location)

    def modify_to_box_coordinates(self, id):
        """Move the origin of the release areas to be correctly displayed in the sandbox"""
        temp = self.release_area_all[int(id) - 1]
        self.release_area = numpy.vstack((temp[:,0]+self.Load_Area.box_origin[0], temp[:,1]+self.Load_Area.box_origin[1])).T

    # Widgets
    def show_widgets(self):
        self._create_widgets()
        tabs = pn.Tabs(('Load and Save Module', self.Load_Area.show_widgets()),
                       ('Landslide simulation module', self.show_landslide_widgets()))
        self.Load_Area._widget_npz_filename.value = self.topo_folder + "Topography1.npz"

        return tabs

    def show_landslide_widgets(self):
        tabs = pn.Tabs(('Controllers', self.widgets_controller_simulation()),
                       ('Load Simulation', self.widgets_load_simulation())
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
        #self._widget_activate_LoadSaveModule = pn.widgets.Button(name='Activate Load Save Module', button_type="success")
        #self._widget_activate_LoadSaveModule.param.watch(self._callback_activate_LoadSaveModule, 'clicks', onlychanged=False)
#
        #self._widget_activate_LandslideModule = pn.widgets.Button(name='Activate Landslide Module',
        #                                                         button_type="success")
        #self._widget_activate_LandslideModule.param.watch(self._callback_activate_LandslideModule, 'clicks',
        #                                                 onlychanged=False)
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
        self.plot_frame_panel()

    def _callback_filename(self, event):
        self.simulation_folder = event.new

    def _callback_load_files_folder(self, event):
        if self.simulation_folder is not None:
            self.load_release_area(self.release_folder)
            self._widget_available_release_areas.options = self.release_id_all
            self._widget_available_release_areas.sizing_mode = "scale_both"

    def _callback_select_frame(self, event):
        self.frame_selector = event.new
        #self.plot_landslide_frame()
        self.plot_frame_panel()

    def _callback_simulation(self, event):
        if event.new == 'Run':
            self.running_simulation = True
        else:
            self.running_simulation = False
        #self.plot_landslide_frame()

    #def _callback_activate_LoadSaveModule(self, event):
     #   #self.Load_Area.setup()
     #   #self.Load_Area.run()
     #   pass

    #def _callback_activate_LandslideModule(self, event):
     #   #self.Load_Area.stop()
     #   #self.setup()
     #   #self.run()
     #   pass

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
