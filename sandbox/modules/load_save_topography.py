import panel as pn
import numpy
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

from .module_main_thread import Module


class LoadSaveTopoModule(Module):
    """
    Module to save the current topography in a subset of the sandbox
    and recreate it at a later time
    two different representations are saved to the numpy file:

    absolute Topography:
    deviation from the mean height inside the bounding box in millimeter

    relative Height:
    height of each pixel relative to the vmin and vmax of the currently used calibration.
    use relative height with the gempy module to get the same geologic map with different calibration settings.
    """

    def __init__(self, *args, **kwargs):
        # call parents' class init, use greyscale colormap as standard and extreme color labeling
        super().__init__(*args, contours=True, cmap='gist_earth_r', over='k', under='k', **kwargs)
        self.box_origin = [40, 40]  #location of bottom left corner of the box in the sandbox. values refer to pixels of the kinect sensor
        self.box_width = 200
        self.box_height = 150
        self.absolute_topo = None
        self.relative_topo = None

        self.is_loaded = False  # Flag to know is a file have been loaded or not
        self.show_loaded = False  # Flag to indicate the axes to be plotted
        self.show_difference = False

        self.difference = None
        self.loaded = None

        self.transparency_difference = 1

        self.cmap_difference = None
        self.norm_difference = None

        self.npz_filename = None

        self.shape_frame = None

        self.release_width = 10
        self.release_height = 10
        self.release_area = None
        self.release_area_origin = None
        self.aruco_release_area_origin = None

        self.snapshot_frame = pn.pane.Matplotlib(plt.figure(), tight=False, height=335)
        plt.close()  # close figure to prevent inline display

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

        if self.show_loaded:
            self.showLoadedTopo()

        if self.show_difference:
            self.showDifference()

        self.showBox(self.box_origin, self.box_width, self.box_height)

        # if aruco Module is specified: update, plot aruco markers
        if self.ARUCO_ACTIVE:
            self.update_aruco()
            self.plot.plot_aruco(self.Aruco.aruco_markers)
            if len(self.Aruco.aruco_markers)> 0:
                self.aruco_release_area_origin = self.Aruco.aruco_markers.loc[
                    self.Aruco.aruco_markers.is_inside_box, ('box_x', 'box_y')]

        self.plot_release_area(self.release_area_origin, self.release_width, self.release_height)
        #self.add_release_area_origin(self.release_area_origin)


        self.projector.trigger()

    def moveBox_possible(self, x, y, width, height):
        if (x+width) >= self.calib.s_frame_width:
            self.box_width = self.calib.s_frame_width - x
        else:
            self.box_width = width

        if (y+height) >= self.calib.s_frame_height:
            self.box_height = self.calib.s_frame_height - y
        else:
            self.box_height = height

        self.box_origin = [x, y]

    def add_release_area_origin(self, x=None, y=None):
        if self.release_area_origin is None:
            self.release_area_origin = pd.DataFrame(columns=(('box_x','box_y')))
        if self.aruco_release_area_origin is None:
            self.aruco_release_area_origin = pd.DataFrame(columns=(('box_x', 'box_y')))
        self.release_area_origin = pd.concat((self.release_area_origin, self.aruco_release_area_origin))
        if x is not None and y is not None:
            self.release_area_origin = self.release_area_origin.append({'box_x': x, 'box_y': y}, ignore_index=True)

    def plot_release_area(self, center, width, height):
        if center is not None:
            x_pos = center.box_x
            y_pos = center.box_y
            x_origin = x_pos.values - width/2
            y_origin = y_pos.values - height/2
            self.release_area = numpy.array([[x_origin-self.box_origin[0], y_origin-self.box_origin[1]],
                                             [x_origin - self.box_origin[0], y_origin+height-self.box_origin[1]],
                                             [x_origin+width - self.box_origin[0], y_origin-self.box_origin[1]],
                                             [x_origin+width - self.box_origin[0], y_origin+height-self.box_origin[1]]])
            for i in range(len(x_pos)):
                self.showBox([x_origin[i], y_origin[i]], width, height)

    def showBox(self, origin, width, height):
        """
        draws a wide rectangle outline in the live view
        :param origin: tuple,relative position from bottom left in sensor pixel space
        :param width: width of box in sensor pixels
        :param height: height of box in sensor pixels

        """
        box = matplotlib.patches.Rectangle(origin, width, height, fill=False, edgecolor='white')
        self.plot.ax.add_patch(box)

    def getBoxFrame(self):
        """
        Get the absolute and relative topo of the sensor readings
        Returns:

        """
        frame = self.sensor.get_filtered_frame()
        if self.crop:
            frame = self.crop_frame(frame)

        # crop sensor image to dimensions of box
        cropped_frame = frame[self.box_origin[1]:self.box_origin[1] + self.box_height,
                        self.box_origin[0]:self.box_origin[0] + self.box_width]

        mean_height = cropped_frame.mean()
        absolute_topo = cropped_frame - mean_height
        relative_topo = absolute_topo / (self.calib.s_max - self.calib.s_min)
        return absolute_topo, relative_topo

    def extractTopo(self):
        self.is_loaded = True
        self.absolute_topo, self.relative_topo = self.getBoxFrame()
        self.shape_frame = self.absolute_topo.shape
        return self.absolute_topo, self.relative_topo

    def saveTopo(self, filename="savedTopography.npz"):
        numpy.savez(filename,
                    self.absolute_topo,
                    self.relative_topo,
                    self.release_area)
        print('Save topo successful')

    def save_release_area(self, filename="releaseArea.npy"):
        numpy.save(filename,
                    self.release_area)
        print('Save area successful')

    def loadTopo(self, filename="savedTopography.npz"):
        self.is_loaded = True
        files = numpy.load(filename, allow_pickle=True)
        self.absolute_topo = files['arr_0']
        self.relative_topo = files['arr_1']
        print('Load successful')

    def showLoadedTopo(self): # Not working
        if self.is_loaded:
            self.getBoxShape()
            self.loaded = self.modify_to_box_coordinates(self.absolute_topo[:self.shape_frame[0],
                                                                            :self.shape_frame[1]])
            self.plot.ax.pcolormesh(self.loaded, cmap='gist_earth_r')
        else:
            print("No Topography loaded, please load a Topography")

    def modify_to_box_coordinates(self, frame):
        width = frame.shape[0]
        left = numpy.ones((self.box_origin[0], width))
        left[left == 1] = numpy.nan
        frame = numpy.insert(frame, 0, left, axis=1)

        height = frame.shape[1]
        bot = numpy.ones((self.box_origin[1], height))
        bot[bot == 1] = numpy.nan
        frame = numpy.insert(frame, 0, bot, axis=0)
        return frame

    def set_cmap(self):
        blues = plt.cm.RdBu(numpy.linspace(0, 0.5, 256))
        reds = plt.cm.RdBu(numpy.linspace(0.5, 1, 256))
        blues_reds = numpy.vstack((blues, reds))
        self.cmap_difference = matplotlib.colors.LinearSegmentedColormap.from_list('difference_map', blues_reds)

    def set_norm(self):
        self.norm_difference = matplotlib.colors.TwoSlopeNorm(vmin=self.absolute_topo.min(),
                                                              vcenter=0,
                                                              vmax=self.absolute_topo.max())

    def getBoxShape(self):
        current_absolute_topo, current_relative_topo = self.getBoxFrame()
        x_dimension, y_dimension = current_absolute_topo.shape
        x_saved, y_saved = self.absolute_topo.shape
        self.shape_frame = [numpy.min((x_dimension, x_saved)), numpy.min((y_dimension, y_saved))]

    def extractDifference(self):
        current_absolute_topo, current_relative_topo = self.getBoxFrame()
        self.getBoxShape()
        diff = self.absolute_topo[:self.shape_frame[0],
                                  :self.shape_frame[1]] - \
               current_absolute_topo[:self.shape_frame[0],
                                     :self.shape_frame[1]]

        # paste diff array at right location according to box coordinates
        self.difference = self.modify_to_box_coordinates(diff)

    def showDifference(self):
        if self.is_loaded:
            self.extractDifference()
            # plot
            self.set_cmap()
            self.set_norm()
            self.plot.ax.pcolormesh(self.difference,
                                    cmap=self.cmap_difference,
                                    alpha=self.transparency_difference,
                                    norm=self.norm_difference)

        else:
            print('No topography to show difference')

    def snapshotFrame(self):
        fig = plt.figure()
        ax = plt.gca()
        ax.cla()
        ax.pcolormesh(self.absolute_topo, cmap='gist_earth_r')
        ax.axis('equal')
        ax.set_axis_off()
        ax.set_title('Loaded Topography')
        self.snapshot_frame.object = fig
        self.snapshot_frame.param.trigger('object')

    def show_widgets(self):
        self._create_widgets()
        tabs = pn.Tabs(('Box widgets', self.widgets_box()),
                       ('Release area widgets', self.widgets_release_area()),
                       ('Save Topography', self.widgets_save()),
                       ('Load Topography', self.widgets_load())
                       )
        return tabs

    def widgets_release_area(self):
        widgets = pn.WidgetBox('<b>Modify the size and shape of the release area </b>',
                               self._widget_release_width,
                               self._widget_release_height,
                               self._widget_show_release)
        panel = pn.Column("### Shape release area", widgets)

        return panel

    def widgets_box(self):
        widgets = pn.WidgetBox('<b>Modify box size </b>',
                               self._widget_move_box_horizontal,
                               self._widget_move_box_vertical,
                               self._widget_box_width,
                               self._widget_box_height,
                               '<b>Take snapshot</b>',
                               self._widget_snapshot,
                               '<b>Show snapshot in sandbox</b>',
                               self._widget_show_snapshot,
                               '<b>Show difference plot</b>',
                               self._widget_show_difference
                               )

        rows = pn.Row(widgets, self.snapshot_frame)
        panel = pn.Column("### Interaction widgets", rows)

        return panel

    def widgets_save(self):
        widgets = pn.WidgetBox('<b>Filename</b>',
                               self._widget_npz_filename,
                               '<b>Safe Topography</b>',
                               self._widget_save,
                               )

        panel = pn.Column("### Save widget", widgets)

        return panel

    def widgets_load(self):
        widgets = pn.WidgetBox('<b>Filename</b>',
                               self._widget_npz_filename,
                               '<b>Load Topography</b>',
                               self._widget_load
                               )

        panel = pn.Column("### Load widget", widgets)

        return panel

    def _create_widgets(self):
        # Box widgets
        self._widget_move_box_horizontal = pn.widgets.IntSlider(name='x box origin',
                                                           value=self.box_origin[0],
                                                           start=0,
                                                           end=self.calib.s_frame_width)
        self._widget_move_box_horizontal.param.watch(self._callback_move_box_horizontal, 'value', onlychanged=False)

        self._widget_move_box_vertical = pn.widgets.IntSlider(name='y box origin',
                                                                value=self.box_origin[1],
                                                                start=0,
                                                                end=self.calib.s_frame_height)
        self._widget_move_box_vertical.param.watch(self._callback_move_box_vertical, 'value', onlychanged=False)

        self._widget_box_width = pn.widgets.IntSlider(name='box width',
                                                              value=self.box_width,
                                                              start=0,
                                                              end=self.calib.s_frame_width)
        self._widget_box_width.param.watch(self._callback_box_width, 'value', onlychanged=False)

        self._widget_box_height = pn.widgets.IntSlider(name='box height',
                                                      value=self.box_height,
                                                      start=0,
                                                      end=self.calib.s_frame_height)
        self._widget_box_height.param.watch(self._callback_box_height, 'value', onlychanged=False)

        # Snapshots
        self._widget_snapshot = pn.widgets.Button(name="Snapshot", button_type="success")
        self._widget_snapshot.param.watch(self._callback_snapshot, 'clicks',
                                                         onlychanged=False)

        # Show snapshots
        self._widget_show_snapshot = pn.widgets.Checkbox(name='Show', value=False)
        self._widget_show_snapshot.param.watch(self._callback_show_snapshot, 'value',
                                                           onlychanged=False)

        self._widget_show_difference = pn.widgets.Checkbox(name='Show', value=False)
        self._widget_show_difference.param.watch(self._callback_show_difference, 'value',
                                               onlychanged=False)

        # Load save widgets
        self._widget_npz_filename = pn.widgets.TextInput(name='Choose a filename for the topography snapshot:')
        self._widget_npz_filename.param.watch(self._callback_filename, 'value', onlychanged=False)
        self._widget_npz_filename.value = 'saved_DEMs/savedTopography.npz'

        self._widget_save = pn.widgets.Button(name='Save')
        self._widget_save.param.watch(self._callback_save, 'clicks', onlychanged=False)

        self._widget_load = pn.widgets.Button(name='Load')
        self._widget_load.param.watch(self._callback_load, 'clicks', onlychanged=False)

        # Release area widgets
        self._widget_release_width = pn.widgets.IntSlider(name='Release area width',
                                                      value=self.release_width,
                                                      start=1,
                                                      end=50)
        self._widget_release_width.param.watch(self._callback_release_width, 'value', onlychanged=False)

        self._widget_release_height = pn.widgets.IntSlider(name='Release area height',
                                                          value=self.release_height,
                                                          start=1,
                                                          end=50)
        self._widget_release_height.param.watch(self._callback_release_height, 'value', onlychanged=False)

        self._widget_show_release = pn.widgets.RadioButtonGroup(name='Show or erase the areas',
                                                              options=['Show', 'Erase'],
                                                              value=['Erase'],
                                                              button_type='success')
        self._widget_show_release.param.watch(self._callback_show_release, 'value', onlychanged=False)

        return True

    def _callback_show_release(self, event):
        if event.new == 'Show':
            self.add_release_area_origin()
        else:
            self.release_area_origin = None

    def _callback_release_width(self, event):
        self.release_width = event.new

    def _callback_release_height(self, event):
        self.release_height = event.new

    def _callback_filename(self, event):
        self.npz_filename = event.new

    def _callback_save(self, event):
        if self.npz_filename is not None:
            self.saveTopo(filename=self.npz_filename)

    def _callback_load(self, event):
        if self.npz_filename is not None:
            self.loadTopo(filename=self.npz_filename)
            self.snapshotFrame()

    def _callback_move_box_horizontal(self, event):
        self.moveBox_possible(x=event.new,
                              y=self.box_origin[1],
                              width=self.box_width,
                              height=self.box_height)

    def _callback_move_box_vertical(self, event):
        self.moveBox_possible(x=self.box_origin[0],
                              y=event.new,
                              width=self.box_width,
                              height=self.box_height)

    def _callback_box_width(self, event):
        self.moveBox_possible(x=self.box_origin[0],
                              y=self.box_origin[1],
                              width=event.new,
                              height=self.box_height)

    def _callback_box_height(self, event):
        self.moveBox_possible(x=self.box_origin[0],
                              y=self.box_origin[1],
                              width=self.box_width,
                              height=event.new)

    def _callback_snapshot(self, event):
        self.extractTopo()
        self.snapshotFrame()

    def _callback_show_snapshot(self, event):
        self.show_loaded = event.new
        self.snapshotFrame()

    def _callback_show_difference(self, event):
        self.show_difference = event.new
        self.snapshotDifference()

    def saveTopoVector(self):
        """
        saves a vector graphic of the contour map to disk
        """
        pass
