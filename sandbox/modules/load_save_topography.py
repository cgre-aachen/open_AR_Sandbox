import os
import panel as pn
import numpy
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

from .template import ModuleTemplate
from sandbox import _test_data
from matplotlib.figure import Figure



class LoadSaveTopoModule(ModuleTemplate):
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
    def __init__(self, extent: list = None, **kwargs):
        # call parents' class init, use greyscale colormap as standard and extreme color labeling
        pn.extension()
        if extent is not None:
            self.vmin = extent[4]
            self.vmax = extent[5]
            self.extent = extent

        self.box_origin = [40, 40]  #location of bottom left corner of the box in the sandbox. values refer to pixels of the kinect sensor
        self.box_width = 200
        self.box_height = 150
        self.absolute_topo = None
        self.relative_topo = None

        self.is_loaded = False  # Flag to know if a file have been loaded or not

        self.current_show = 'None'
        self.difference_types = ['None', 'Show topography', 'Show difference', 'Show gradient difference']

        self.cmap_difference = self._cmap_difference()

        self.difference = None
        self.loaded = None

        self.transparency_difference = 1

        self.npz_filename = None

        self.release_width = 10
        self.release_height = 10
        self.release_area = None
        self.release_area_origin = None
        self.aruco_release_area_origin = None

        self.data_filenames = ['None']
        self.file_id = None
        self.data_path = _test_data['topo']

        self.figure = Figure()
        self.ax = plt.Axes(self.figure, [0., 0., 1., 1.])
        self.figure.add_axes(self.ax)

        self.snapshot_frame = pn.pane.Matplotlib(self.figure, tight=False, height=500)
        plt.close(self.figure)  # close figure to prevent inline display

        #Stores the axes
        self._lod = None
        #self._dif = None
        self.frame = None
        return print("LoadSaveTopoModule loaded succesfully")

    def update(self, sb_params: dict):
        frame = sb_params.get('frame')
        ax = sb_params.get('ax')
        marker = sb_params.get('marker')
        self.frame = frame
        if len(marker) > 0:
            self.aruco_release_area_origin = marker.loc[marker.is_inside_box, ('box_x', 'box_y')]
            self.add_release_area_origin()
        self.plot(frame, ax)

        return sb_params

    def delete_rectangles_ax(self, ax):
        [rec.remove() for rec in reversed(ax.patches) if isinstance(rec, matplotlib.patches.Rectangle)]
        #ax.patches = []

    def delete_im_ax(self, ax):
        #[quad.remove() for quad in reversed(ax.collections) if isinstance(quad, matplotlib.collections.QuadMesh)]
        #if self._dif is not None:
        #    self._dif.remove()
        #    self._dif = None
        if self._lod is not None:
            self._lod.remove()
            self._lod = None

    def set_show(self, i: str):
        self.current_show = i

    def plot(self, frame, ax):
        self.delete_rectangles_ax(ax)
        self.delete_im_ax(ax)
        if self.current_show == self.difference_types[0]:
            self.delete_im_ax(ax)
        elif self.current_show == self.difference_types[1]:
            self.showLoadedTopo(ax)
        elif self.current_show == self.difference_types[2]:
            self.showDifference(ax)
        elif self.current_show == self.difference_types[3]:
            self.showGradDifference(ax)

        self.showBox(ax, self.box_origin, self.box_width, self.box_height)
        self.plot_release_area(ax, self.release_area_origin, self.release_width, self.release_height)

    def moveBox_possible(self, x, y, width, height):
        """
        Dinamicaly modify the size of the box when the margins extend more than frame
        Args:
            x: x coordinte of the box origin
            y: y coordinate of the box origin
            width: of the box
            height: of the box
        Returns:
        """

        if (x+width) >= self.extent[1]:
            self.box_width = self.extent[1] - x
        else:
            self.box_width = width

        if (y+height) >= self.extent[3]:
            self.box_height = self.extent[3] - y
        else:
            self.box_height = height

        self.box_origin = [x, y]

    def add_release_area_origin(self, x=None, y=None):
        """
        Add a box origin [x,y] to highlight a zone on the image.
        This method also manages the aruco release areas if detected
        Args:
            x: x coordinte of origin
            y: y coordinte of origin
        Returns:

        """
        if self.release_area_origin is None:
            self.release_area_origin = pd.DataFrame(columns=(('box_x','box_y')))
        if self.aruco_release_area_origin is None:
            self.aruco_release_area_origin = pd.DataFrame(columns=(('box_x', 'box_y')))
        self.release_area_origin = pd.concat((self.release_area_origin, self.aruco_release_area_origin)).drop_duplicates()
        if x is not None and y is not None:
            self.release_area_origin = self.release_area_origin.append({'box_x': x, 'box_y': y}, ignore_index=True)

    def plot_release_area(self, ax, origin: pd.DataFrame, width: int, height: int):
        """
        Plot small boxes in the frame according to the dataframe origin and width, height specifiend.
        Args:
            ax: matplotlib axes to plot
            origin: pandas dataframe indicating the x and y coordintes of the boxes to plot
            width: width of the box to plot
            height: height of the box to plot
        Returns:
        """
        if origin is not None:
            x_pos = origin.box_x
            y_pos = origin.box_y
            x_origin = x_pos.values - width/2
            y_origin = y_pos.values - height/2
            self.release_area = numpy.array([[x_origin-self.box_origin[0], y_origin-self.box_origin[1]],
                                             [x_origin - self.box_origin[0], y_origin+height-self.box_origin[1]],
                                             [x_origin+width - self.box_origin[0], y_origin+height-self.box_origin[1]],
                                             [x_origin+width - self.box_origin[0], y_origin-self.box_origin[1]]])
            for i in range(len(x_pos)):
                self.showBox(ax, [x_origin[i], y_origin[i]], width, height)

    def showBox(self, ax, origin: tuple, width: int, height: int):
        """
        Draws a wide rectangle outline in the live view
        Args:
            ax: the axes where the patch will be drawed on
            origin: relative position from bottom left in sensor pixel space
            width: width of box in sensor pixels
            height: height of box in sensor pixels
        Returns:
        """
        box = matplotlib.patches.Rectangle(origin, width, height, fill=False, edgecolor='white')
        ax.add_patch(box)
        return True

    def getBoxFrame(self, frame: numpy.ndarray):
        """
        Get the absolute and relative topography of the current.
        Crop frame image to dimensions of box
        Args:
            frame: frame of the actual topography
        Returns:
            absolute_topo, the cropped frame minus the mean value and relative_topo, the absolute topo normalized to the extent of the sandbox
        """
        cropped_frame = frame[self.box_origin[1]:self.box_origin[1] + self.box_height,
                        self.box_origin[0]:self.box_origin[0] + self.box_width]

        mean_height = cropped_frame.mean()
        absolute_topo = cropped_frame - mean_height
        relative_topo = absolute_topo / (self.vmax - self.vmin)
        return absolute_topo, relative_topo

    def extractTopo(self):
        """
        Extract the topography of the current frame and stores the value internally
        Returns:
            absolute topography and relative topography
        """
        self.is_loaded = True
        self.absolute_topo, self.relative_topo = self.getBoxFrame(self.frame)
        return self.absolute_topo, self.relative_topo

    def saveTopo(self, filename="00_savedTopography.npz"):
        """Save the absolute topography and relative topography in a .npz file"""
        numpy.savez(filename,
                    self.absolute_topo,
                    self.relative_topo)
        print('Save topo successful')

    def save_release_area(self, filename="00_releaseArea.npy"):
        """Save the release areas as a .npy file """
        numpy.save(filename,
                    self.release_area)
        print('Save area successful')

    def loadTopo(self, filename="00_savedTopography.npz"):
        """Load the absolute topography and relative topography from a .npz file"""
        self.is_loaded = True
        files = numpy.load(filename, allow_pickle=True)
        self.absolute_topo = files['arr_0']
        self.relative_topo = files['arr_1']
        print('Load successful')
        self._get_id(filename)

    def showLoadedTopo(self, ax):
        """
        If a topography is saved internally, display the saved topograhy on the sandbox
        Args:
            ax: axes to plot the saved topography
        Returns:
        """
        if self.is_loaded:
            shape_frame = self.getBoxShape()
            #self.loaded = self.modify_to_box_coordinates(self.absolute_topo[:shape_frame[0],
            #                                             :shape_frame[1]])
            self.loaded = self.absolute_topo[:shape_frame[0], :shape_frame[1]]
            #if self._lod is None:

            self._lod = ax.imshow(self.loaded, cmap='gist_earth', origin="lower", #TODO: data is inverted, need to fix this for all the landsladides topography data
                                      zorder=2, extent=self.to_box_extent, aspect="auto")
            #else:
             #   self._lod.set_array(self.loaded[:-1,:-1].ravel())
        else:
          #  if self._lod is not None:
           #     self._lod.remove()
           #     self._lod = None
            print("No Topography loaded, please load a Topography")

    def modify_to_box_coordinates(self, frame):
        """
        Since the box is not in the origin of the frame,
        this will correctly display the loaded topography inside the box
        Args:
            frame: the frame that need to be modified to box coordintes
        Returns:
            The modified frame
        """
        width = frame.shape[0]
        left = numpy.ones((self.box_origin[0], width))
        left[left == 1] = numpy.nan
        frame = numpy.insert(frame, 0, left, axis=1)

        height = frame.shape[1]
        bot = numpy.ones((self.box_origin[1], height))
        bot[bot == 1] = numpy.nan
        frame = numpy.insert(frame, 0, bot, axis=0)
        #frame = numpy.ma.array(frame, mask=numpy.nan)
        return frame

    def saveTopoVector(self):  # TODO:
        """
        saves a vector graphic of the contour map to disk
        """
        pass

    def _cmap_difference(self):
        """Creates a custom made color map"""
        blues = plt.cm.RdBu(numpy.linspace(0, 0.5, 256))
        reds = plt.cm.RdBu(numpy.linspace(0.5, 1, 256))
        blues_reds = numpy.vstack((blues, reds))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('difference_map', blues_reds)
        return cmap

    @property
    def norm_difference(self):
        """Creates a custom made norm"""
        norm = matplotlib.colors.TwoSlopeNorm(vmin=self.absolute_topo.min(),
                                                         vcenter=0,
                                                         vmax=self.absolute_topo.max())
        return norm

    def getBoxShape(self):
        """This will return the shape of the current saved topography"""
        current_absolute_topo, current_relative_topo = self.getBoxFrame(self.frame)
        x_dimension, y_dimension = current_absolute_topo.shape
        x_saved, y_saved = self.absolute_topo.shape
        shape_frame = [numpy.min((x_dimension, x_saved)), numpy.min((y_dimension, y_saved))]
        return shape_frame

    def extractDifference(self):
        """This will return a numpy array comparing the difference between the current frame and the saved frame """
        current_absolute_topo, _ = self.getBoxFrame(self.frame)
        shape_frame = self.getBoxShape()
        diff = self.absolute_topo[:shape_frame[0],
                                  :shape_frame[1]] - \
               current_absolute_topo[:shape_frame[0],
                                     :shape_frame[1]]

        # paste diff array at right location according to box coordinates
        #difference = self.modify_to_box_coordinates(diff)
        return diff

    @property
    def to_box_extent(self):
        """When using imshow to plot data over the image. pass this as extent argumment to display the
        image in the correct area of the sandbox box-area"""
        return (self.box_origin[0], self.box_width+self.box_origin[0],
                self.box_origin[1], self.box_height+self.box_origin[1])

    def showDifference(self, ax):
        """
        Displays the calculated difference of the previous frame with the actual frame
        Args:
            ax: Axes to plot the difference
        Returns:
        """
        if self.is_loaded:
            difference = self.extractDifference()
            # plot
           # if self._dif is None:
            self._lod = ax.imshow(difference,
                                            cmap=self.cmap_difference,
                                            alpha=self.transparency_difference,
                                            norm=self.norm_difference,
                                       origin = "lower",
                                       zorder=1,
                                   extent  =self.to_box_extent,
                                  aspect="auto"
                                       )
            #else:
             #   self._dif.set_array(difference[:-1, :-1].ravel())
        else:
            #if self._dif is not None:
            #    self._dif.remove()
             #   self._dif = None
            print('No topography to show difference')

    def showGradDifference(self, ax):
        """
        Displays the calculated gradient difference of the previous frame with the actual frame
        Args:
            ax: Axes to plot the difference
        Returns:
        """
        if self.is_loaded:
            grad = self.extractGradDifference()
            # plot
            # if self._dif is None:
            self._lod = ax.imshow(grad,
                                  vmin=-5,
                                  vmax=5,
                                  cmap=self.cmap_difference,
                                  alpha=self.transparency_difference,
                                  norm=self.norm_difference,
                                  origin="lower",
                                  zorder=1,
                                  extent=self.to_box_extent,
                                  aspect="auto"
                                  )
            # else:
            #   self._dif.set_array(difference[:-1, :-1].ravel())
        else:
            # if self._dif is not None:
            #    self._dif.remove()
            #   self._dif = None
            print('No topography to show gradient difference')

    def extractGradDifference(self):
        """This will return a numpy array comparing the difference of second degree (gradients)
        between the current frame and the saved frame """
        current_absolute_topo, _ = self.getBoxFrame(self.frame)
        dx_current, dy_current = numpy.gradient(current_absolute_topo)
        dxdy_current = numpy.sqrt(dx_current ** 2 + dy_current ** 2)
        dxdy_current = numpy.clip(dxdy_current, -5, 5)

        dx_lod, dy_lod = numpy.gradient(self.absolute_topo)
        dxdy_lod = numpy.sqrt(dx_lod ** 2 + dy_lod ** 2)
        dxdy_lod = numpy.clip(dxdy_lod, -5, 5)

        shape_frame = self.getBoxShape()
        gradDiff = dxdy_current[:shape_frame[0],
                   :shape_frame[1]] - \
               dxdy_lod[:shape_frame[0],
               :shape_frame[1]]

        # paste diff array at right location according to box coordinates
        # difference = self.modify_to_box_coordinates(diff)
        return gradDiff*-1

    def snapshotFrame(self):
        """This will display the saved topography and display it in the panel bokeh"""
        self.ax.cla()
        self.ax.imshow(self.absolute_topo, cmap='gist_earth', origin = "lower", aspect='auto')
        self.ax.axis('equal')
        self.ax.set_axis_off()
        self.ax.set_title('Loaded Topography')
        self.snapshot_frame.param.trigger('object')

    def _search_all_data(self, data_path):
        self.data_filenames = os.listdir(data_path)

    def _get_id(self, filename):
        ids = [str(s) for s in filename if s.isdigit()]
        if len(ids) > 0:
            self.file_id = ids[-1]
        else:
            print("Unknown file id")

    def show_widgets(self):
        tabs = pn.Tabs(('Box widgets', self.widgets_box()),
                       ('Release area widgets', self.widgets_release_area()),
                       ('Load Topography', self.widgets_load()),
                       ('Save Topography', self.widgets_save())
                       )
        return tabs

    def widgets_release_area(self):
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

        widgets = pn.WidgetBox('<b>Modify the size and shape of the release area </b>',
                               self._widget_release_width,
                               self._widget_release_height,
                               self._widget_show_release)
        panel = pn.Column("### Shape release area", widgets)

        return panel

    def widgets_box(self):
        # Box widgets
        self._widget_show_type = pn.widgets.RadioBoxGroup(name='Show in sandbox',
                                                          options=self.difference_types,
                                                          value=self.difference_types[0],
                                                          inline=False)
        self._widget_show_type.param.watch(self._callback_show, 'value', onlychanged=False)

        self._widget_move_box_horizontal = pn.widgets.IntSlider(name='x box origin',
                                                                value=self.box_origin[0],
                                                                start=0,
                                                                end=self.extent[1])
        self._widget_move_box_horizontal.param.watch(self._callback_move_box_horizontal, 'value', onlychanged=False)

        self._widget_move_box_vertical = pn.widgets.IntSlider(name='y box origin',
                                                              value=self.box_origin[1],
                                                              start=0,
                                                              end=self.extent[3])
        self._widget_move_box_vertical.param.watch(self._callback_move_box_vertical, 'value', onlychanged=False)

        self._widget_box_width = pn.widgets.IntSlider(name='box width',
                                                      value=self.box_width,
                                                      start=0,
                                                      end=self.extent[1])
        self._widget_box_width.param.watch(self._callback_box_width, 'value', onlychanged=False)

        self._widget_box_height = pn.widgets.IntSlider(name='box height',
                                                       value=self.box_height,
                                                       start=0,
                                                       end=self.extent[3])
        self._widget_box_height.param.watch(self._callback_box_height, 'value', onlychanged=False)

        # Snapshots
        self._widget_snapshot = pn.widgets.Button(name="Snapshot", button_type="success")
        self._widget_snapshot.param.watch(self._callback_snapshot, 'clicks',
                                          onlychanged=False)

        widgets = pn.Column('<b>Modify box size </b>',
                               self._widget_move_box_horizontal,
                               self._widget_move_box_vertical,
                               self._widget_box_width,
                               self._widget_box_height,
                               '<b>Take snapshot</b>',
                               self._widget_snapshot,
                               '<b>Show in sandbox</b>',
                            self._widget_show_type
                               #self._widget_show_snapshot,
                               #'<b>Show difference plot</b>',
                               #self._widget_show_difference
                               )

        rows = pn.Row(widgets, self.snapshot_frame)
        panel = pn.Column("### Interaction widgets", rows)

        return panel

    def widgets_save(self):
        self._widget_npz_filename = pn.widgets.TextInput(
            name='Choose a filename to save the current topography snapshot:')
        self._widget_npz_filename.param.watch(self._callback_filename_npz, 'value', onlychanged=False)
        self._widget_npz_filename.value = _test_data['topo'] + '/savedTopography.npz'

        self._widget_save = pn.widgets.Button(name='Save')
        self._widget_save.param.watch(self._callback_save, 'clicks', onlychanged=False)

        panel = pn.Column("### Save widget",
                          '<b>Filename</b>',
                          self._widget_npz_filename,
                          '<b>Safe Topography</b>',
                          self._widget_save
                          )
        return panel

    def widgets_load(self):
        self._widget_data_path = pn.widgets.TextInput(
            name='Choose a folder to load the available topography snapshots:')
        self._widget_data_path.value = self.data_path
        self._widget_data_path.param.watch(self._callback_filename, 'value', onlychanged=False)

        self._widget_load = pn.widgets.Button(name='Load Files in folder')
        self._widget_load.param.watch(self._callback_load, 'clicks', onlychanged=False)

        self._widget_available_topography = pn.widgets.RadioBoxGroup(name='Available Topographies',
                                                                     options=self.data_filenames,
                                                                     inline=False)
        self._widget_available_topography.param.watch(self._callback_available_topography, 'value',
                                                      onlychanged=False)

        # self._widget_other_topography = pn.widgets.FileInput(name="Load calibration (Note yet working)")
        self._widget_other_topography = pn.widgets.FileSelector('~')
        # self._widget_other_topography.param.watch(self._callback_other_topography, 'value')
        self._widget_load_other = pn.widgets.Button(name='Load other', button_type='success')
        self._widget_load_other.param.watch(self._callback_load_other, 'clicks', onlychanged=False)

        panel = pn.Column("### Load widget",
                          '<b>Directory path</b>',
                          self._widget_data_path,
                          '<b>Load Topography</b>',
                          self._widget_load,
                          '<b>Load available Topography</b>',
                          pn.WidgetBox(self._widget_available_topography),
                          '<b>Select another Topography file</b>',
                          self._widget_other_topography,
                          self._widget_load_other
                          )

        return panel

    def _callback_show(self, event):
        self.set_show(event.new)

    def _callback_show_release(self, event):
        if event.new == 'Show':
            self.add_release_area_origin()
        else:
            self.release_area_origin = None

    def _callback_release_width(self, event):
        self.release_width = event.new

    def _callback_release_height(self, event):
        self.release_height = event.new

    def _callback_filename_npz(self, event):
        self.npz_filename = event.new

    def _callback_filename(self, event):
        self.data_path = event.new

    def _callback_save(self, event):
        if self.npz_filename is not None:
            self.saveTopo(filename=self.npz_filename)

    def _callback_load(self, event):
        if self.data_path is not None:
            #self.loadTopo(filename=self.npz_filename)
            #self.snapshotFrame()
            self._search_all_data(data_path=self.data_path)
            self._widget_available_topography.options = self.data_filenames
            self._widget_available_topography.sizing_mode = "scale_both"

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


    def _callback_available_topography(self, event):
        if event.new is not None:
            self.loadTopo(filename=self.data_path+event.new)
            self.snapshotFrame()

    def _callback_load_other(self, event):
        self.loadTopo(filename=self._widget_other_topography.value[0])
        self.snapshotFrame()
