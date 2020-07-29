from warnings import warn
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
import numpy
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn' # TODO: SettingWithCopyWarning appears when using LoadTopoModule with arucos
import panel as pn
import threading
import pyvista as pv
from sandbox.modules.module_main_thread import Module

from .utils import get_scale, Grid
from .plot import plot_gempy

#from sandbox.projector.plot import Plot

try:
    import gempy
    from gempy.core.grid_modules.topography import Topography
    from gempy.core.grid_modules import section_utils
except ImportError:
    warn('gempy package not found, GempyModule will not work')


class GemPyModule(Module):
    def __init__(self, *args, geo_model = None, extent: list = None, box: list = None, ** kwargs):
        # TODO: include save elevation map and export geologic map --self.geo_map
        """

        Args:
            geo_model:
            grid:
            geol_map:
            work_directory:

        Returns:
            None

        """
        self.geo_model = geo_model
        try:
            self._model_extent = self.geo_model._grid.regular_grid.extent
        except:
            print('Geo model not valid')
            raise AttributeError
        self._sensor_extent = extent
        self._box_dimensions = box
        self.frame = None
        self.cmap = None
        self.grid = None
        self.model_dict = None
        self.plot_topography = True
        self.plot_faults = True
        self.cross_section = None
        self.section_dict = None
        self.resolution_section = [150, 100]
        self.figsize = (10, 10)
        self.section_traces = None
        self.geological_map = None
        self.section_actual_model = None

        # Manage panel figure to show current model
        self._figure_actual_model = Figure()
        self.ax_actual_model = plt.Axes(self._figure_actual_model, [0., 0., 1., 1.])
        self._figure_actual_model.add_axes(self.ax_actual_model)
        self.ax_actual_model.set_axis_off()
        self.fig_actual_model = pn.pane.Matplotlib(self._figure_actual_model, tight=False, height=500)
        plt.close(self._figure_actual_model)

        # Manage panel figure to show 2D plots ( Cross-sections or geological maps)
        self.fig_plot_2d = pn.pane.Matplotlib(plt.figure(), tight=False, height=335)
        plt.close()

        self.section_dict_boreholes = {}
        self.borehole_tube = []
        self.colors_bh = []
        self.radius_borehole = 20

        #dataframe to safe Arucos in model Space:
        self.modelspace_arucos = pd.DataFrame()

        dummy_frame = numpy.ones((self._sensor_extent[3], self._sensor_extent[1])) * 100
        self.setup(dummy_frame)


    def setup(self, frame):
        self.scale = get_scale(physical_extent=self._box_dimensions,
                           sensor_extent=self._sensor_extent,
                           model_extent=self.geo_model._grid.regular_grid.extent)  # prepare the scale object

        self.grid = Grid(physical_extent=self._box_dimensions,
                           sensor_extent=self._sensor_extent,
                         model_extent=self.geo_model._grid.regular_grid.extent,
                         scale=self.scale)
        self.init_topography(frame)

    def init_topography(self, frame):
        self.grid.update_grid(frame)
        self.geo_model._grid.topography = Topography(self.geo_model._grid.regular_grid)
        self.geo_model._grid.topography.extent = self.grid.model_extent[:4]
        self.geo_model._grid.topography.resolution = numpy.asarray((self.grid.sensor_extent[3], self.grid.sensor_extent[1]))
        self.geo_model._grid.topography.values = self.grid.depth_grid
        self.geo_model._grid.topography.values_3D = numpy.dstack(
            [self.grid.depth_grid[:, 0].reshape(self.grid.sensor_extent[3], self.grid.sensor_extent[1]),
             self.grid.depth_grid[:, 1].reshape(self.grid.sensor_extent[3], self.grid.sensor_extent[1]),
             self.grid.depth_grid[:, 2].reshape(self.grid.sensor_extent[3], self.grid.sensor_extent[1])])

        self.geo_model._grid.set_active('topography')
        self.geo_model.update_from_grid()

    def update(self, frame, ax, extent, marker=[], **kwargs):
        self.frame = frame #Store the current frame
        self.grid.update_grid(frame)
        self.geo_model._grid.topography.values = self.grid.depth_grid
        data = self.grid.depth_grid[:, 2].reshape(self.geo_model._grid.topography.resolution)
        self.geo_model._grid.topography.values_3D[:, :, 2] = data
        self.geo_model._grid.update_grid_values()
        self.geo_model.update_from_grid()

        gempy.compute_model(self.geo_model, compute_mesh=False)
        # if self.marker:
        #    self.compute_modelspace_arucos()
        #    self.plot.plot_aruco(self.modelspace_arucos)
        # self.get_section_dict(self.modelspace_arucos)
        ax, cmap = self.plot(ax, self.geo_model)
        norm = None
        return frame, ax, extent, cmap, norm

    def plot(self, ax, geo_model):
        ax, cmap = plot_gempy(ax, geo_model)
        return ax, cmap

    def change_model(self, geo_model):
        self.geo_model = geo_model
        self.setup(self.frame)
        return True

    def cross_section(self):
        self.get_section_dict(self.modelspace_arucos, "cross_section")

    def boreholes(self):
        pass

    def get_section_dict(self, df, mode): # TODO: Change here
        if len(df) > 0:
            df.sort_values('box_x', ascending=True)
            x = df.box_x.values
            y = df.box_y.values
            self.section_dict = {'aruco_section': ([x[0], y[0]], [x[1], y[1]], self.resolution_section)}

    def _plot_section_traces(self):
        self.geo_model.set_section_grid(self.section_dict)
        self.section_traces = gempy._plot.plot_section_traces(self.geo_model)

    def plot_section_traces(self):
        self.geo_model.set_section_grid(self.section_dict)
        self.section_traces = gempy.plot.plot_section_traces(self.geo_model)

    def plot_cross_section(self):
        self.geo_model.set_section_grid(self.section_dict)
        self.cross_section = gempy.plot_2d(self.geo_model,
                                                 section_names=['aruco_section'],
                                                 figsize=self.figsize,
                                                 show_topography=True,
                                                 show_data=False)

    def plot_geological_map(self):
        self.geological_map = gempy.plot_2d(self.geo_model,
                                                  section_names=['topography'],
                                                  show_data=False,
                                                  figsize=self.figsize)

    def plot_actual_model(self, name):
        self.geo_model.set_section_grid({'section:' + ' ' + name: ([0, 500], [1000, 500], self.resolution_section)})
        _ = gempy.compute_model(self.geo_model, compute_mesh=False)
        self.section_actual_model = gempy.plot_2d(self.geo_model,
                                                        section_names=['section:' + ' ' + name],
                                                        show_data=False,
                                                        figsize=self.figsize)

    def compute_modelspace_arucos(self):
        df = self.Aruco.aruco_markers.copy()
        if len(df) > 0:

            df = df.loc[df.is_inside_box, ('box_x', 'box_y', 'is_inside_box')]
            #df['box_z'] = self.Aruco.aruco_markers.loc[self.Aruco.aruco_markers.is_inside_box, ['Depth_Z(mm)']]
            df['box_z'] = numpy.nan
            # depth is changing all the time so the coordinate map method becomes old.
            # Workaround: just replace the value from the actual frame
            frame = self.crop_frame(self.sensor.depth)


            for i in df.index:
                df.at[i, 'box_z'] = (self.scale.extent[5] -
                                     ((frame[int(df.at[i, 'box_x'])][(int(df.at[i, 'box_y']))] - self.calib.s_min) /
                                      (self.calib.s_max - self.calib.s_min) *
                                      (self.scale.extent[5] - self.scale.extent[4])))
                #the combination below works though it should not! Look into scale again!!
                #pixel scale and pixel size should be consistent!
                df.at[i, 'box_x'] = (self.scale.pixel_size[0]*self.Aruco.aruco_markers['box_x'][i])
                df.at[i, 'box_y'] = (self.scale.pixel_scale[1]*self.Aruco.aruco_markers['box_y'][i])

        self.modelspace_arucos = df

    def borehole_cross_section(self, df):
        if len(df) > 0:
            bh = {}
            for i in df.index:
                point1 = numpy.array([df.loc[i, 'box_x'], df.loc[i, 'box_y']])
                point2 = numpy.array([df.loc[i, 'box_x']+1, df.loc[i, 'box_y']])
                bh.update({'id_'+str(i): ([point1[0], point1[1]], [point2[0], point2[1]], [5,5])})

            self.section_dict_boreholes = bh

    def get_polygon_data(self):
        self.borehole_tube = []
        self.colors_bh = []
        _ = self.geo_model.set_section_grid(self.section_dict_boreholes)
        _ = gempy.compute_model(self.geo_model, compute_mesh=False)
        for index_id in self.modelspace_arucos.index:
            polygondict, cdict, extent = section_utils.get_polygon_dictionary(self.geo_model,
                                                                              section_name="id_"+str(index_id))
            plt.close()  # avoid inline display

            point_bh = numpy.array([self.modelspace_arucos.loc[index_id, 'box_z'], self.scale.extent[4]]) #top and bottom of model
            colors_bh = numpy.array([])  # to save the colors of the model
            for form in cdict.keys():
                pointslist = numpy.array(polygondict[form])
                start_point = pointslist[0][0][0]
                c = cdict[form]
                colors_bh = numpy.append(colors_bh, c)
                if pointslist.shape != ():
                    for points in pointslist:
                        x = points[:, 0]
                        y = points[:, 1]
                        val = y[x == start_point].min()
                        point_bh = numpy.append(point_bh, val)

            point_bh[::-1].sort()

            closed = point_bh[point_bh > self.modelspace_arucos.loc[index_id, 'box_z']]
            point_bh = point_bh[point_bh <= self.modelspace_arucos.loc[index_id, 'box_z']]
            print(closed, colors_bh)
            colors_bh = colors_bh[:len(point_bh)-1][::-1]

            x_val = numpy.ones(len(point_bh)) * self.modelspace_arucos.loc[index_id, 'box_x']
            y_val = numpy.ones(len(point_bh)) * self.modelspace_arucos.loc[index_id, 'box_y']
            z_val = point_bh

            borehole_points = numpy.vstack((x_val, y_val, z_val)).T

            line = self.lines_from_points(borehole_points)
            print(borehole_points, colors_bh)
            line["scalars"] = numpy.arange(len(colors_bh)+1)

            self.borehole_tube.append(line.tube(radius=self.radius_borehole))
            self.colors_bh.append(colors_bh)

    def plot_boreholes(self):
        p = pv.Plotter(notebook=False)
        for i in range(len(self.borehole_tube)):
            cmap = self.colors_bh[i]
            p.add_mesh(self.borehole_tube[i], cmap=[cmap[j] for j in range(len(cmap))])
        extent = numpy.copy(self.scale.extent)
        extent[-1] = numpy.ceil(self.modelspace_arucos.box_z.max()/100)*100
        p.show_bounds(bounds=extent)
        p.show()

    def lines_from_points(self, points):
        """Given an array of points, make a line set"""
        poly = pv.PolyData()
        poly.points = points
        cells = numpy.full((len(points) - 1, 3), 2, dtype=numpy.int)
        cells[:, 1] = numpy.arange(0, len(points) - 1, dtype=numpy.int)
        cells[:, 2] = numpy.arange(1, len(points), dtype=numpy.int)
        poly.lines = cells
        return poly


    def show_widgets(self):
        tabs = pn.Tabs(('Cross_sections', self.widget_plot2d()),
                       ("Boreholes"),
                       ('Plot', self.widget_plot_module())
                       )

        return tabs

    def widget_model_selector(self, Model_dict):
        self.model_dict = Model_dict
        pn.extension()
        self._widget_model_selector = pn.widgets.RadioButtonGroup(name='Model selector',
                                                                  options=list(self.model_dict.keys()),
                                                                  value=list(self.model_dict.keys())[0],
                                                                  button_type='success')
        self._widget_model_selector.param.watch(self._callback_selection, 'value', onlychanged=False)

        widgets = pn.WidgetBox(self._widget_model_selector,
                               self.fig_actual_model,
                               width=550
                               )

        panel = pn.Column("### Model Selector widgets", widgets)
        return panel

    def widget_plot2d(self):
        self._create_widgets()
        widgets = pn.WidgetBox('<b>Create a cross section</b>',
                               self._widget_select_plot2d,
                               self.fig_plot_2d
                               )
        panel = pn.Column('### Creation of 2D Plots', widgets)
        return panel

    def _create_widgets(self):
        pn.extension()
        self._widget_select_plot2d = pn.widgets.RadioButtonGroup(name='Plot 2D',
                                             options=['Geological_map', 'Section_traces', 'Cross_Section'],
                                             value=['Geological_map'],
                                             button_type='success')
        self._widget_select_plot2d.param.watch(self._callback_selection_plot2d, 'value', onlychanged=False)

    def _callback_selection(self, event): # TODO: Not working properly, change in notebook
        """
        callback function for the widget to update the self.
        :return:
        """
        self.stop()
        geo_model = self.model_dict[event.new]
        self.change_model(geo_model)
        #self.plot_actual_model(event.new)
        #self.fig_actual_model.object = self.section_actual_model.fig
        #self.fig_actual_model.object.param.trigger('object')

    def _callback_selection_plot2d(self, event):
        if event.new == 'Geological_map':
            self.plot_geological_map()
            self.fig_plot_2d.object = self.geological_map.fig
            self.fig_plot_2d.object.param.trigger('object')
        elif event.new == 'Section_traces':
            self.plot_section_traces()
            self.fig_plot_2d.object = self.section_traces.fig
            self.fig_plot_2d.object.param.trigger('object')
        elif event.new == 'Cross_Section':
            self.plot_cross_section()
            self.fig_plot_2d.object = self.cross_section.fig
            self.fig_plot_2d.object.param.trigger('object')
