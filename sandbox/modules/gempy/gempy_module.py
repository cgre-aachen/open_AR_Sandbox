from warnings import warn
import matplotlib.pyplot as plt
import numpy
import panel as pn

from sandbox.modules.template import ModuleTemplate
from sandbox.modules.gempy.utils import get_scale, Grid
from sandbox.modules.gempy.plot import plot_gempy
from sandbox.modules.gempy.example_models import create_model_dict, all_models

import pandas as pd
# TODO: SettingWithCopyWarning appears when using LoadTopoModule with arucos
pd.options.mode.chained_assignment = None  # default='warn'

try:
    import pyvista as pv
    import gempy
    from gempy.core.grid_modules.topography import Topography
    from gempy.core.grid_modules import section_utils
except ImportError:
    warn('gempy package not found, GempyModule will not work')


class GemPyModule(ModuleTemplate):
    def __init__(self, geo_model=None, extent: list = None, box: list = None,
                 load_examples: bool = True, name_example: list = all_models, ** kwargs) -> object:
        # TODO: include save elevation map and export geologic map --self.geo_map
        """

        Args:
            geo_model: Previously constructed geo_model ready for visualization
            extent: sensor extents
            box: physical extends of the sandbox
            load_examples: To load all the example models and switch between them using a dictionary
        Returns:
            None

        """
        pn.extension('vtk')  # TODO: check if all the usages of extensions are actually changing something
        self.lock = None  # For locking the multithreading while using bokeh server
        if load_examples and len(name_example) > 0:
            self.model_dict = create_model_dict(name_example, **kwargs)
            print("Examples loaded in dictionary model_dict")
        else: self.model_dict = None

        if geo_model is None and self.model_dict is not None:
            self.geo_model = self.model_dict[name_example[0]]
            print("Model " + name_example[0] + " loaded as geo_model")
        else:
            self.geo_model = geo_model
            if self.model_dict is None:
                self.model_dict = {}
            self.model_dict[geo_model.meta.project_name] = geo_model

        try:
            self._model_extent = self.geo_model._grid.regular_grid.extent
        except:
            print('Geo model not valid')
            raise AttributeError
        self._sensor_extent = extent
        self._box_dimensions = box

        self.frame = None
        self.vmin = None
        self.vmax = None
        self.cmap = None
        self.grid = None
        self.plot_topography = True
        self.plot_faults = True
        self.cross_section = None
        self.section_dict = {}
        self.borehole_dict = {}
        self.actual_dict = {}
        self._resolution_section = [150, 100]
        self.figsize = (10, 10)

        # 2D images
        self.im_section_traces = None
        self.im_plot_2d = None
        self.im_actual_model = None
        self.im_geo_map = None

        # 3D gempy model
        self.geo_3d = None

        self._plotter_type = 'basic'
        self._notebook = False
        self._param_3d_model = {'show_data': True,
                                'show_results': True,
                                'show_surfaces': True,
                                'show_lith': True,
                                'show_scalar': False,
                                'show_boundaries': True,
                                'show_topography': False}
        self._ve = 0.3

        # Manage panel figure to show current model
        self.panel_section_traces = pn.pane.Matplotlib(plt.figure(), tight=False, height=500)
        plt.close()
        # Manage panel figure to show 2D plots ( Cross-sections or geological maps)
        self.panel_plot_2d = pn.pane.Matplotlib(plt.figure(), tight=False, height=500)
        plt.close()

        self.panel_actual_model = pn.pane.Matplotlib(plt.figure(), tight=False, height=500)
        plt.close()

        self.panel_geo_map = pn.pane.Matplotlib(plt.figure(), tight=False, height=500)
        plt.close()

        p1 = pv.Plotter(notebook=False)
        self.vtk_borehole = pn.panel(p1.ren_win,
                                     sizing_mode='stretch_both',
                                     orientation_widget=True,
                                     enable_keybindings=True)
        p2 = pv.Plotter(notebook=False)
        self.vtk_model = pn.panel(p2.ren_win,
                                  sizing_mode='stretch_both',
                                  orientation_widget=True,
                                  enable_keybindings=True)

        # For the boreholes potting
        self.borehole_tube = []
        self.colors_bh = []
        self.faults_bh = []
        self.faults_color_bh = []
        self._radius_borehole = 20

        # For the new plotting way 'TODO: Create widgets
        self.show_lith = True
        self.show_boundary = True
        self.show_hillshades = False
        self.show_contour = False
        self.show_only_faults = False
        self.show_fill_contour = False

        # dataframe to save Arucos in model Space:
        self.modelspace_arucos = pd.DataFrame()

        dummy_frame = numpy.ones((self._sensor_extent[3], self._sensor_extent[1])) * 1000
        self.setup(dummy_frame)

    def setup(self, frame):
        self.frame = frame
        self.vmin = frame.min()
        self.vmax = frame.max()
        self._scale, self._pixel_scale, self._pixel_size = get_scale(
            physical_extent=self._box_dimensions,
            sensor_extent=self._sensor_extent,
            model_extent=self.geo_model._grid.regular_grid.extent)  # prepare the scale object

        self.grid = Grid(physical_extent=self._box_dimensions,
                         sensor_extent=self._sensor_extent,
                         model_extent=self.geo_model._grid.regular_grid.extent,
                         scale=self._scale)
        self.init_topography(frame)

    def init_topography(self, frame):
        self.grid.update_grid(frame)
        self.geo_model._grid.topography = Topography(self.geo_model._grid.regular_grid)
        self.geo_model._grid.topography.extent = self.grid.model_extent[:4]
        self.geo_model._grid.topography.resolution = numpy.asarray((self.grid.sensor_extent[3], self.grid.sensor_extent[1]))
        self.geo_model._grid.topography.values = self.grid.depth_grid
        self.geo_model._grid.topography.values_2d = numpy.dstack(
            [self.grid.depth_grid[:, 0].reshape(self.grid.sensor_extent[3], self.grid.sensor_extent[1]),
             self.grid.depth_grid[:, 1].reshape(self.grid.sensor_extent[3], self.grid.sensor_extent[1]),
             self.grid.depth_grid[:, 2].reshape(self.grid.sensor_extent[3], self.grid.sensor_extent[1])])

        self.geo_model._grid.set_active('topography')
        self.geo_model.update_from_grid()
        self.set_actual_dict()

    def update(self, sb_params: dict):
        frame = sb_params.get('frame')
        extent = sb_params.get('extent')
        ax = sb_params.get('ax')
        marker = sb_params.get('marker')
        self.lock = sb_params.get('lock_thread')
        self.frame = frame  # Store the current frame
        self.vmin = frame.min()
        self.vmax = frame.max()
        scale_frame = self.grid.scale_frame(frame)
        _ = self.grid.update_grid(scale_frame)
        self.geo_model._grid.topography.values = self.grid.depth_grid
        data = self.grid.depth_grid[:, 2].reshape(self.geo_model._grid.topography.resolution)
        self.geo_model._grid.topography.values_2d[:, :, 2] = data
        _= self.geo_model._grid.update_grid_values()
        _= self.geo_model.update_from_grid()

        gempy.compute_model(self.geo_model, compute_mesh=False)
        if len(marker) > 0:
            self.modelspace_arucos = self._compute_modelspace_arucos(marker)
            self.set_aruco_dict(self.modelspace_arucos)

        ax, cmap = self.plot(ax, self.geo_model, self._model_extent)

        sb_params['ax'] = ax
        sb_params['frame'] = scale_frame
        sb_params['cmap'] = cmap
        sb_params['marker'] = self.modelspace_arucos
        sb_params['active_cmap'] = False
        sb_params['active_shading'] = False
        sb_params['extent'] = self._model_extent
        sb_params['del_contour'] = not self.show_boundary

        return sb_params

    def plot(self, ax, geo_model, extent):
        # ax, cmap = plot_gempy_topography(ax, geo_model, extent,
        ax, cmap = plot_gempy(ax, geo_model, extent,
                              show_lith=self.show_lith,
                              show_boundary=self.show_boundary,
                              show_hillshade=self.show_hillshades,
                              show_contour=self.show_contour,
                              show_only_faults=self.show_only_faults,
                              show_fill_contour=self.show_fill_contour)
        return ax, cmap

    def change_model(self, geo_model):
        """
        Change a gempy model
        Args:
            geo_model: New gempy model to replace
        Returns:
        """
        self.remove_section_dict('Model: ' + self.geo_model.meta.project_name)
        self.geo_model = geo_model
        self.setup(self.frame)
        print("New gempy model loaded")
        return True

    @property
    def model_sections_dict(self):
        """One time calculation to join dictionaries needed for cross_sections and boreholes"""
        return {**self.section_dict, **self.borehole_dict, **self.actual_dict}

    def _compute_modelspace_arucos(self, marker):
        """Receive a dataframe with the location of the arucos and then conver it to model coordinates.
        Args:
            marker: dataframe with aruco locations
        Returns:
            new dataframe with scaled values
        """
        df = marker.copy()
        if len(df) > 0:
            df = df.loc[df.is_inside_box, ('box_x', 'box_y', 'is_inside_box')]
            # df['box_z'] = self.Aruco.aruco_markers.loc[self.Aruco.aruco_markers.is_inside_box, ['Depth_Z(mm)']]
            df['box_z'] = numpy.nan
            # depth is changing all the time so the coordinate map method becomes old.
            # Workaround: just replace the value from the actual frame
            frame = self.frame
            for i in df.index:
                df.at[i, 'box_z'] = self.grid.scale_frame(frame[int(df.at[i, 'box_y'])][(int(df.at[i, 'box_x']))])
                # the combination below works though it should not! Look into scale again!!
                # pixel scale and pixel size should be consistent!
                df.at[i, 'box_x'] = (self._pixel_scale[0]*marker['box_x'][i])
                df.at[i, 'box_y'] = (self._pixel_scale[1]*marker['box_y'][i])

        return df

    def set_aruco_dict(self, df):
        """
        Receive an aruco dataframe already in the model coordinates and set a cross_section and points for the borehole
        Args:
            df: aruco dataframe

        Returns:
            change in place the section dictionary
        """
        if len(df) > 0:
            # include boreholes
            for i in df.index:
                x = df.at[i, 'box_x']
                y = df.at[i, 'box_y']
                self.set_borehole_dict((x,y), "aruco_"+str(i))
            if len(df) == 2:
                # Obtain the position of the aruco markers (must be 2 aruco markers)
                # to draw a cross-section by updating the section dictionary
                df.sort_values('box_x', ascending=True)
                x = df.box_x.values
                y = df.box_y.values
                p1 = (x[0], y[0])
                p2 = (x[1], y[1])
                self.set_section_dict(p1, p2, "Aruco_section")

    def set_section_dict(self, p1, p2, name):
        """
        Actualize the section dictionary to draw the cross_sections by appending he new points
        Args:
            p1: Point 1 (x,y) coordinates. The most left one
            p2: To point 2 (x,y) coordinates. The most right one
            name: Name of the section dictionary
        Returns:
            change in place the section dictionary
        """
        self.section_dict[name] = ([p1[0], p1[1]], [p2[0], p2[1]], self._resolution_section)
        _ = self.geo_model.set_section_grid(self.model_sections_dict)
        _ = gempy.compute_model(self.geo_model, compute_mesh=False)

    def set_actual_dict(self):
        """
        Actualize the section dictionary to draw the cross_sections of the actual model
        Returns
            change in place the actual dictionary
        """
        self.actual_dict = {}
        self.actual_dict['Model: ' + self.geo_model.meta.project_name] = ([self._model_extent[0],
                                                                           self._model_extent[3]/2],
                                                                          [self._model_extent[1],
                                                                           self._model_extent[3]/2],
                                                                          self._resolution_section)
        _ = self.geo_model.set_section_grid(self.model_sections_dict)
        _ = gempy.compute_model(self.geo_model, compute_mesh=False)

    def remove_section_dict(self, name: str):
        """
        Remove a specific section
        Args:
            name: Key name
        Returns:
        """
        if name in self.section_dict.keys():
            self.section_dict.pop(name)
            _ = self.geo_model.set_section_grid(self.model_sections_dict)
            _ = gempy.compute_model(self.geo_model, compute_mesh=False)
        else:
            print("No key found with name ", name, " in section_dict")

    def _get_aruco_section_dict(self, df):
        """Obtain the position of the aruco markers (must be 2 aruco markers)
        to draw a cross-section by updating the section dictionary"""
        if len(df) > 0:
            df.sort_values('box_x', ascending=True)
            x = df.box_x.values
            y = df.box_y.values
            p1 = (x[0], y[0])
            p2 = (x[1], y[1])
            self.set_section_dict(p1, p2, "Aruco_section")

    def show_section_traces(self):
        """Show the current location in the sandbox where the cross-section is painted"""
        self.im_section_traces = gempy.plot.plot_section_traces(self.geo_model)
        plt.close()
        self.panel_section_traces.object = self.im_section_traces.fig
        self.panel_section_traces.param.trigger('object')
        return self.im_section_traces.fig

    def show_geological_map(self):
        """Show the geological map from the gempy package"""
        self.im_geo_map = gempy.plot_2d(self.geo_model, section_names=['topography'], show_data=False,
                                        show_topography=True, show=False)
        self.panel_geo_map.object = self.im_geo_map.fig
        self.panel_geo_map.param.trigger('object')
        return self.im_geo_map.fig

    def show_cross_section(self, name: str):
        """
        Show the 2d cross_section or geological map
        Args:
            name: Show the cross section of the
        Returns:
        """
        if name in self.section_dict.keys():
            self.im_plot_2d = gempy.plot_2d(self.geo_model, section_names=[name], show_data=False, show_topography=True,
                                            show=False)
            # self.im_plot_2d.axes[0].set_ylim(self.frame.min(), self.frame.max())
            self.im_plot_2d.axes[0].set_aspect(aspect=0.5)
            self.panel_plot_2d.object = self.im_plot_2d.fig
            self.panel_plot_2d.param.trigger('object')
            return self.im_plot_2d.fig
        else: print("no key in section_dict have the name: ", name)

    def show_actual_model(self):
        """Show a cross_section of the actual gempy model"""
        # Get a cross_section in the middle of the model
        self.set_actual_dict()
        self.im_actual_model = gempy.plot_2d(self.geo_model,
                                             section_names=['Model: ' + self.geo_model.meta.project_name],
                                             show_data=False,
                                             show=False,
                                             show_topography=False)
        # self.im_actual_model.axes[0].set_ylim(self.frame.min(), self.frame.max())
        self.im_actual_model.axes[0].set_aspect(aspect=0.5)
        self.panel_actual_model.object = self.im_actual_model.fig
        self.panel_actual_model.param.trigger('object')
        return self.im_actual_model.fig

    def _get_aruco_borehole_dict(self, df):
        """Obtain the position of the aruco markers to update the borehole dictionary"""
        if len(df) > 0:
            # Search in the dataframe for new markers to add or update
            for i in df.index:
                point1 = numpy.array([df.loc[i, 'box_x'], df.loc[i, 'box_y']])
                point2 = numpy.array([df.loc[i, 'box_x'] + 1, df.loc[i, 'box_y']])
                self.borehole_dict['id_'+str(i)] = ([point1[0], point1[1]], [point2[0], point2[1]], [5, 5])
            # after adding the new markers, check for markers that dont exist
            # anymore and remove them from the dictionary
            for i in self.borehole_dict.keys():
                temp = df.loc[df.index == int(i[-1])].index
                if len(temp) > 0 and temp[0] == int(i[-1]):
                    pass
                else:
                    self.remove_borehole_dict(name=i)

    def remove_borehole_dict(self, name: str):
        """
        Remove a specific borehole dict
        Args:
            name: Key name
        Returns:
        """
        if name in self.borehole_dict.keys():
            self.borehole_dict.pop(name)
            _ = self.geo_model.set_section_grid(self.model_sections_dict)
            _ = gempy.compute_model(self.geo_model, compute_mesh=False)
        else:
            print("No key found with name ", name, " in borehole_dict")

    def set_borehole_dict(self, xy, name):
        """
        Actualize the section dictionary to draw the cross_sections by appending he new points
        Args:
            xy: Point 1 xy[0] coordinates. The most left one To point 2 xy[1] coordinates. The most right one
            name: Name of the section dictionary
        Returns:
            change in place the section dictionary
        """
        self.borehole_dict[name] = ([xy[0], xy[1]], [xy[0]+1, xy[1]], [5, 5])
        _ = self.geo_model.set_section_grid(self.model_sections_dict)
        _ = gempy.compute_model(self.geo_model, compute_mesh=False)

    def _get_polygon_data(self):
        """
        Method that gets the polygondict, cdict and extent of all the borehole points and store them in lines and colors
        """

        self.borehole_tube = []
        self.colors_bh = []
        self.faults_bh = []
        self.faults_color_bh = []
        _ = self.geo_model.set_section_grid(self.model_sections_dict)
        _ = gempy.compute_model(self.geo_model, compute_mesh=False)
        faults = list(self.geo_model.surfaces.df.loc[self.geo_model.surfaces.df['isFault']]['surface'])
        for name in self.borehole_dict.keys():
            polygondict, cdict, extent = section_utils.get_polygon_dictionary(self.geo_model,
                                                                              section_name=name)
            plt.close()  # avoid inline display

            # To get the top point of the model
            x, y = self.borehole_dict[name][0][0], self.borehole_dict[name][0][1]
            _ = self.grid.scale_frame(self.frame[int(y/self._pixel_scale[1]), int(x/self._pixel_scale[0])])
            z = numpy.asarray([_, _])
            color = numpy.asarray([None])
            fault_point = numpy.asarray([])
            fault_color = numpy.asarray([])
            for formation in list(self.geo_model.surfaces.df['surface']):
                 for path in polygondict.get(formation):
                     if path != []:
                        vertices = path.vertices
                        _idx = (numpy.abs(vertices[:, 0] - extent[1]/2)).argmin()
                        _compare = vertices[:, 0][_idx]
                        _mask = numpy.where(vertices[:, 0] == _compare)
                        extremes = vertices[_mask]
                        z_val = extremes[:, 1]
                        if formation in faults:
                            # fault_point = numpy.append(fault_point, z_val)
                            # fault_color = numpy.append(fault_color, cdict.get(formation))
                            self.faults_bh.append(numpy.asarray([x, y, z_val[0]]))
                            self.faults_color_bh.append(cdict.get(formation))
                        else:
                            z = numpy.vstack((z, z_val))
                            color = numpy.append(color, cdict.get(formation))

            mask1 = z[:, 0].argsort()
            mask2 = z[:, 0][mask1] <= z[0, 0]  # This is the first value added to start counting

            z_final = z[:, 0][mask1][mask2]
            color_final = color[mask1][mask2]
            # color_final[-1] = color[mask1][mask2 == False][0] Not needed to replace the color since is already none

            x_final = numpy.ones(len(z_final)) * x
            y_final = numpy.ones(len(z_final)) * y

            borehole_points = numpy.vstack((x_final, y_final, z_final)).T

            line = self._lines_from_points(borehole_points)
            line["scalars"] = numpy.arange(len(color_final))

            # For a single borehole
            self.borehole_tube.append(line.tube(radius=self._radius_borehole))
            self.colors_bh.append(color_final)
            # if len(fault_point) > 0:
            #    self.faults_bh.append(numpy.asarray([x, y, fault_point]))
            #    self.faults_color_bh.append(fault_color)

    def _lines_from_points(self, points):
        """Given an array of points, make a line set.
        See https://docs.pyvista.org/examples/00-load/create-spline.html
        for more information
        Args:
            points: x,y,z coordinates of the points
        """
        poly = pv.PolyData()
        poly.points = points
        cells = numpy.full((len(points) - 1, 3), 2, dtype=numpy.int)
        cells[:, 1] = numpy.arange(0, len(points) - 1, dtype=numpy.int)
        cells[:, 2] = numpy.arange(1, len(points), dtype=numpy.int)
        poly.lines = cells
        return poly

    def plot_boreholes(self, notebook=False, background=False, **kwargs):
        """
        Uses the previously calculated borehole tubes in self._get_polygon_data()
        when a borehole dictionary is available
        This will generate a pyvista object that can be visualized with .show()
        Args:
            notebook: If using in notebook to show inline
            background:
        Returns:
            Pyvista object with all the boreholes
        """
        self._get_polygon_data()
        if background:
            p = pv.BackgroundPlotter(**kwargs)
        else:
            p = pv.Plotter(notebook=notebook, **kwargs)
        for i in range(len(self.borehole_tube)):
            cmap = self.colors_bh[i]
            p.add_mesh(self.borehole_tube[i], cmap=[cmap[j] for j in range(len(cmap)-1)], smooth_shading=False)
        # for i in range(len(self.faults_bh)):
        # for plotting the faults
        # TODO: Messing with the colors when faults
        if len(self.faults_bh) > 0:
            point = pv.PolyData(self.faults_bh)
            p.add_mesh(point, render_points_as_spheres=True, point_size=self._radius_borehole)
            # p.add_mesh(point, cmap = self.faults_color_bh[i],
            # render_points_as_spheres=True, point_size=self._radius_borehole)
        extent = numpy.copy(self._model_extent)
        # extent[-1] = numpy.ceil(self.modelspace_arucos.box_z.max()/100)*100
        p.show_bounds(bounds=extent)
        p.show_grid()
        p.set_scale(zscale=self._ve)
        # self.vtk = pn.panel(p.ren_win, sizing_mode='stretch_width', orientation_widget=True)
        # self.vtk = pn.Row(pn.Column(pan, pan.construct_colorbars()), pn.pane.Str(type(p.ren_win), width=500))
        return p

    def show_boreholes_panel(self):
        """This function will show the pyvista object of plot_boreholes in a panel server"""
        pl = self.plot_boreholes(notebook = False)
        pan = pn.panel(pl.ren_win, orientation_widget=True, enable_keybindings=True, sizing_mode='scale_both')
        axes = dict(
            origin=[self._model_extent[0], self._model_extent[2], self._model_extent[4]],
            xticker={'ticks': numpy.linspace(self._model_extent[0], self._model_extent[1], 5)},
            yticker={'ticks': numpy.linspace(self._model_extent[2], self._model_extent[3], 5)},
            zticker={'ticks': numpy.linspace(self._model_extent[4], self._model_extent[5], 5),
                     'labels': [''] + [str(int(item)) for item in numpy.linspace(self._model_extent[4],
                                                                                 self._model_extent[5], 5)[1:]]},
            fontsize=12,
            digits=1,
            grid_opacity=0.5,
            show_grid=True)
        pan.axes = axes
        widget = pn.Row(pn.Column(pan, pan.construct_colorbars()), pn.pane.Str(type(pl.ren_win)))  # , width=500))

        self.vtk_borehole = widget
        # self.vtk.object = pan.object
        # self.vtk.param.trigger('object')
        return self.vtk_borehole

    def plot_3d_model(self):
        """Generate a 3D gempy model and return a the pyvista object"""
        self.geo_3d = gempy.plot_3d(self.geo_model,
                                    plotter_type=self._plotter_type,
                                    show_data=self._param_3d_model['show_data'],
                                    show_results=self._param_3d_model['show_results'],
                                    show_surfaces=self._param_3d_model['show_surfaces'],
                                    show_lith=self._param_3d_model['show_lith'],
                                    show_scalar=self._param_3d_model['show_scalar'],
                                    show_boundaries=self._param_3d_model['show_boundaries'],
                                    show_topography=self._param_3d_model['show_topography'],
                                    notebook=self._notebook,
                                    image=False,
                                    off_screen=False,
                                    ve=self._ve
                                    )
        return self.geo_3d

    def show_3d_model_panel(self): #TODO: NOT WORKING
        """This function will show the pyvista object of plot_3d_model in a panel server"""
        pl = self.plot_3d_model()
        pan = pn.panel(pl.p.ren_win, width=700, sizing_mode='stretch_both', orientation_widget=True,
                       enable_keybindings=True)
        axes = dict(
            origin=[self._model_extent[0], self._model_extent[2], self._model_extent[4]],
            xticker={'ticks': numpy.linspace(self._model_extent[0], self._model_extent[1], 5)},
            yticker={'ticks': numpy.linspace(self._model_extent[2], self._model_extent[3], 5)},
            zticker={'ticks': numpy.linspace(self._model_extent[4], self._model_extent[5], 5),
                     'labels': [''] + [str(int(item)) for item in
                                       numpy.linspace(self._model_extent[4], self._model_extent[5], 5)[1:]]},
            fontsize=12,
            digits=1,
            grid_opacity=0.5,
            show_grid=True)
        pan.axes = axes
        widget = pn.Row(pn.Column(pan, pan.construct_colorbars()), pn.pane.Str(type(pl.ren_win)))  # , width=500))

        self.vtk_model = widget
        return self.vtk_model

    # Panel widgets
    def show_widgets(self):
        _ = self.show_actual_model()
        tabs = pn.Tabs(('Models', self.widget_model_selector()),
                       ('Geological map', self.widget_geological_map()),
                       ('Section traces', self.widget_section_traces()),
                       ('Cross_sections', self.widget_cross_sections()),
                       ('Boreholes', self.widget_boreholes()),
                       ('3D Gempy Model', self.widget_3d_model())
                       )
        return tabs

    def widget_3d_model(self):
        self._widget_show_3d_model = pn.widgets.Button(name="Show 3D Gempy Model", button_type="success",
                                                       disabled=True)
        # TODO: Fix this
        self._widget_show_3d_model.param.watch(self._callback_show_3d_model, 'clicks', onlychanged=False)
        self._widget_show_3d_model_pyvista = pn.widgets.Button(name="Show 3D Gempy Model pyvista",
                                                               button_type="warning")
        self._widget_show_3d_model_pyvista.param.watch(self._callback_show_3d_model_pyvista, 'clicks',
                                                       onlychanged=False)

        self._widget_parameters_3d_model = pn.widgets.CheckBoxGroup(name='Select properties to show of gempy model',
                                                                    options=list(self._param_3d_model.keys()),
                                                                    value=[active for active
                                                                           in self._param_3d_model.keys()
                                                                           if self._param_3d_model[active] == True],
                                                                    inline=False)

        self._widget_parameters_3d_model.param.watch(self._callback_param_3d_model, 'value', onlychanged=False)
        self._widget_vertical_exageration = pn.widgets.Spinner(name='Vertical Exaggeration',value=self._ve, step=0.1)
        self._widget_vertical_exageration.param.watch(self._callback_vertical_exageration, 'value',
                                                      onlychanged=False)

        widgets = pn.Column('### Show 3D Gempy Model',
                            self._widget_show_3d_model,
                            self._widget_show_3d_model_pyvista,
                            '<b>Select properties to show of gempy model</b>',
                            self._widget_parameters_3d_model,
                            self._widget_vertical_exageration)
        return widgets

    def widget_boreholes(self):
        self._widget_show_boreholes = pn.widgets.Button(name="Show Boreholes panel", button_type="success")
        self._widget_show_boreholes.param.watch(self._callback_show_boreholes, 'clicks',
                                                onlychanged=False)

        self._widget_show_boreholes_pyvista = pn.widgets.Button(name="Show Boreholes pyvista", button_type="warning")
        self._widget_show_boreholes_pyvista.param.watch(self._callback_show_boreholes_pyvista, 'clicks',
                                                        onlychanged=False)
        self._w_borehole_name = pn.widgets.TextInput(name='Borehole name', value='BH_1')
        self._w_x = pn.widgets.TextInput(name='x:', value='10.0', width=60)
        self._w_y = pn.widgets.TextInput(name='y:', value='20.0', width=60)

        self._widget_add_bh = pn.widgets.Button(name="Add borehole", button_type="success")
        self._widget_add_bh.param.watch(self._callback_add_bh, 'clicks',
                                        onlychanged=False)

        self._w_remove_borehole_name = pn.widgets.AutocompleteInput(name='Remove borehole name',
                                                                    options=list(self.borehole_dict.keys()))
        self._widget_remove_bh = pn.widgets.Button(name="Remove borehole", button_type="success")
        self._widget_remove_bh.param.watch(self._callback_remove_bh, 'clicks',
                                           onlychanged=False)

        self._widget_boreholes_available = pn.widgets.RadioBoxGroup(name='Available boreholes',
                                                                     options=list(self.borehole_dict.keys()),
                                                                     inline=False,
                                                                    disabled=True
                                                                     )

        widgets = pn.Column('### Creation of boreholes',
                            self._widget_show_boreholes,
                            self._widget_show_boreholes_pyvista,
                            '<b>add new borehole </b>',
                            pn.WidgetBox(self._w_borehole_name,
                                         pn.Row(self._w_x, self._w_y)),
                            self._widget_add_bh,
                            '<b>Remove borehole</b>',
                            self._w_remove_borehole_name,
                            self._widget_remove_bh,
                            '<b>Loaded boreholes</b>',
                            self._widget_boreholes_available,
                            )
        # TODO: add method to include more boreholes

        return widgets
    
    def widget_geological_map(self):
        self._widget_update_geo_map = pn.widgets.Button(name="Update Geological map", button_type="success")
        self._widget_update_geo_map.param.watch(self._callback_geo_map, 'clicks',
                                                onlychanged=False)
        widget = pn.Column("### Geological Map",
                           self._widget_update_geo_map, self.panel_geo_map)
        # TODO: add save geological map here. Maybe include vector map
        return widget

    def widget_section_traces(self):
        self._widget_update_section_traces = pn.widgets.Button(name="Update Section Traces", button_type="success")
        self._widget_update_section_traces.param.watch(self._callback_section_traces, 'clicks',
                                                onlychanged=False)
        widget = pn.Column("### Section Traces",
                           self._widget_update_section_traces, self.panel_section_traces)
        # TODO: add widgets to add or remove cross_sections
        return widget

    def widget_cross_sections(self):
        self._widget_select_cross_section = pn.widgets.RadioBoxGroup(name='Available Cross sections',
                                                                     options=list(self.section_dict.keys()),
                                                                     inline=False
                                                                     )
        
        self._widget_select_cross_section.param.watch(self._callback_selection_plot2d, 'value', onlychanged=False)

        self._widget_update_cross_section = pn.widgets.Button(name="Update Cross Section", button_type="success")
        self._widget_update_cross_section.param.watch(self._callback_cross_section, 'clicks', onlychanged=False)

        self._w_section_name = pn.widgets.TextInput(name="Name cross section:", value='CS_1')
        self._w_p1_x = pn.widgets.TextInput(name='x:', value= '10.0', width=60)
        self._w_p1_y = pn.widgets.TextInput(name='y:', value= '20.0', width=60)

        self._w_p2_x = pn.widgets.TextInput(name='x:', value='200.0', width=60)
        self._w_p2_y = pn.widgets.TextInput(name='y:', value='400.0', width=60)

        self._widget_add_cs = pn.widgets.Button(name="Add cross section", button_type="success")
        self._widget_add_cs .param.watch(self._callback_add_cs, 'clicks', onlychanged=False)

        self._w_remove_name = pn.widgets.AutocompleteInput(name='Cross section name',
                                                           options=list(self.section_dict.keys()))

        self._widget_remove_cs = pn.widgets.Button(name="Remove cross section", button_type="success")
        self._widget_remove_cs.param.watch(self._callback_remove_cs, 'clicks', onlychanged=False)

        widgets = pn.Column('### Creation of 2D Plots',
                            self._widget_update_cross_section,
                            '<b>add new cross section</b>',
                            pn.WidgetBox(self._w_section_name,
                                         pn.Row(pn.WidgetBox('From',
                                                             self._w_p1_x,
                                                             self._w_p1_y,
                                                             horizontal=True),
                                                pn.WidgetBox('To',
                                                             self._w_p2_x,
                                                             self._w_p2_y,
                                                             horizontal=True))),
                            self._widget_add_cs,
                            '<b>Remove cross section</b>',
                            self._w_remove_name,
                            self._widget_remove_cs,
                            '<b>Select cross section to display</b>',
                            self._widget_select_cross_section,
                            )

        panel = pn.Row(widgets, self.panel_plot_2d)
        return panel

    def widget_model_selector(self):
        self._widget_model_selector = pn.widgets.RadioButtonGroup(name='Model selector',
                                                                  options=list(self.model_dict.keys()),
                                                                  value=self.geo_model.meta.project_name,
                                                                  button_type='success')
        self._widget_model_selector.param.watch(self._callback_selection, 'value', onlychanged=False)

        panel = pn.Column("### Model Selector widgets",
                            self._widget_model_selector,
                            self.panel_actual_model)

        return panel

    def _callback_add_cs(self, event):
        name = self._w_section_name.value
        p1 = (float(self._w_p1_x.value), float(self._w_p1_y.value))
        p2 = (float(self._w_p2_x.value), float(self._w_p2_y.value))
        self.set_section_dict(p1, p2, name)
        self._widget_select_cross_section.options = list(self.section_dict.keys())
        self._widget_remove_cs.options = list(self.section_dict.keys())

    def _callback_remove_cs(self, event):  # TODO: Not working properly
        self.remove_section_dict(self._w_remove_name.value)
        self._widget_select_cross_section.options = list(self.section_dict.keys())

    def _callback_add_bh(self, event):
        name = self._w_borehole_name.value
        xy = (float(self._w_x.value), float(self._w_x.value))
        self.set_borehole_dict(xy, name)
        self._widget_boreholes_available.options = list(self.borehole_dict.keys())
        self._widget_remove_bh.options = list(self.borehole_dict.keys())

    def _callback_remove_bh(self, event):  # TODO: Not working properly
        self.remove_section_dict(self._w_remove_borehole_name.value)
        self._widget_boreholes_available.options = list(self.borehole_dict.keys())
        self._w_remove_borehole_name.options = list(self.section_dict.keys())

    def _callback_param_3d_model(self, event):
        for key in self._param_3d_model.keys():
            if key in event.new:
                self._param_3d_model[key] = True
            else:
                self._param_3d_model[key] = False

    def _callback_show_3d_model(self, event):
        self.lock.acquire()
        vtk = self.show_3d_model_panel()
        vtk.show()
        self.lock.release()

    def _callback_show_3d_model_pyvista(self, event):
        self.lock.acquire()
        geo = self.plot_3d_model()
        geo.p.show()
        self.lock.release()

    def _callback_show_boreholes_pyvista(self, event):
        self.lock.acquire()
        p = self.plot_boreholes(notebook=False)
        p.show()
        self.lock.release()

    def _callback_section_traces(self, event):
        self.lock.acquire()
        _ = self.show_section_traces()
        self.lock.release()

    def _callback_geo_map(self, event):
        self.lock.acquire()
        _ = self.show_geological_map()
        self.lock.release()

    def _callback_cross_section(self, event):
        self.lock.acquire()
        _ = self.show_cross_section(self._widget_select_cross_section.value)
        self.lock.release()

    def _callback_selection(self, event):
        """
        callback function for the widget to update the self.
        :return:
        """
        self.lock.acquire()
        geo_model = self.model_dict[event.new]
        self.change_model(geo_model)
        self.lock.release()

    def _callback_show_boreholes(self, event):
        self._get_polygon_data()
        vtk = self.show_boreholes_panel()
        vtk.show()

    def _callback_vertical_exageration(self, event):
        self._ve = event.new

    def _callback_selection_plot2d(self, event):
        _ = self.show_cross_section(event.new)
