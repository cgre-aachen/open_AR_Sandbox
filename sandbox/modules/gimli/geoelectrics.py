import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import panel as pn
import numpy
import pygimli as pg
import pygimli.meshtools as mt
import pygimli.physics.ert as ert
from pygimli.viewer.mpl import drawStreams
import logging                                                                                                                                                             
pg.setLogLevel(logging.WARNING)  # to reduce logging output

from ..template import ModuleTemplate


class GeoelectricsModule(ModuleTemplate):
    """
    Module to show the potential fields of a geoelectric (DC resistivity) forward modelling and sesitivity analysis
    check "https://www.pygimli.org/_examples_auto/3_dc_and_ip/plot_04_ert_2_5d_potential.html#sphx-glr-examples-auto-3-dc-and-ip-plot-04-ert-2-5d-potential-py"
    for more information
    """
    def __init__(self, *args, extent: list = None, **kwargs):
        if extent is not None:
            self.vmin = extent[4]
            self.vmax = extent[5]
            self.extent = extent

        self.mesh_fine = None
        self.mesh = None
        self.data_fine = None
        self.data = None
        self.scheme = None
        self.sim = None
        self.pot = None
        self.fop = None
        self.normsens = None
        self.electrode = None

        self.step = 7.5
        self.sensitivity = False
        self.view = "mesh"  # "potential"  "sensitivity""
        self.id = None
        self.real_time = False
        self.p_stream = False
        self.frame = None
        self._df_markers = None
        self._figsize = (15, 8)

        #Widgets
        self._a_id = 0
        self._b_id = 1
        self._m_id = 2
        self._n_id = 3
        self.figure_mesh = Figure()
        self.axes_mesh = Axes(self.figure_mesh, [0., 0., 1., 1.])
        self.figure_mesh.add_axes(self.axes_mesh)
        self.panel_figure_mesh = pn.pane.Matplotlib(self.figure_mesh, tight=True)
        plt.close(self.figure_mesh)  # close figure to prevent inline display

        self.figure_field = Figure()
        self.axes_field = Axes(self.figure_field, [0., 0., 1., 1.])
        self.figure_field.add_axes(self.axes_field)
        self.panel_figure_field = pn.pane.Matplotlib(self.figure_field, tight=True)
        plt.close(self.figure_field)  # close figure to prevent inline display

        self.figure_sensitivity = Figure()
        self.axes_sensitivity = Axes(self.figure_sensitivity, [0., 0., 1., 1.])
        self.figure_sensitivity.add_axes(self.axes_sensitivity)
        self.panel_figure_sensitivity = pn.pane.Matplotlib(self.figure_sensitivity, tight=True)
        plt.close(self.figure_sensitivity)  # close figure to prevent inline display

        return print("GeoelectricsModule loaded successfully")
        
    def update(self, sb_params: dict):
        frame = sb_params.get('frame')
        extent = sb_params.get('extent')
        frame, extent = self.normalize(frame, extent)
        ax = sb_params.get('ax')
        markers = sb_params.get('marker')
        self._df_markers = markers
        self.lock = sb_params.get("lock_thread")
        if len(markers)>0 and len(markers.loc[markers.is_inside_box]) == 4 and self.id is not None:
            self.set_aruco_electrodes(markers)
            if self.real_time:
                self.update_resistivity(frame, self.extent, self.step)
                if self.sensitivity:
                    self.calculate_sensitivity()

        ax = self.plot(ax, self.extent)
        sb_params["frame"] = frame
        sb_params["extent"] = extent
        return sb_params

    def plot(self, ax, extent):
        self.delete_image(ax)
        if self.p_stream and self.pot is not None:
            self.plot_stream_lines(ax)
        if self.view == "potential" and self.pot is not None:
            self.plot_potential(ax, extent)
        elif self.view == "sensitivity" and self.normsens is not None:
            self.plot_sensitivity(ax, extent)
        return ax

    def normalize(self, frame, extent):
        """
        Modify the frame to get some realistic Ohm*m values (frame + 5)* 100
        # TODO: Normalize logaritmically to have more extreme values
        Args:
            frame: sandbox frame
            extent: sandbox extent
        Returns:
            modifyied sandbox frame and extent(vmin and vmax)
        """
        self.frame = (frame + 5) * 100
        self.extent = extent
        self.extent[-2] = (self.extent[-2] + 5) * 100  # vmin
        self.extent[-1] = (self.extent[-1] + 5) * 100  # vmax
        return self.frame, self.extent

    def delete_image(self, ax):
        #[coll.remove() for coll in reversed(ax.collections) if isinstance(coll, matplotlib.collections.PathCollection )]
        ax.cla() #TODO: find a better way to do this 
    def plot_stream_lines(self, ax):
        drawStreams(ax, self.mesh, -self.pot, color='Black')
    def plot_mesh(self, ax, frame, cmap):
        ax.imshow(frame, origin="lower", cmap=cmap)
    def plot_potential(self, ax, extent):
        pg.show(self.mesh, self.pot, ax=ax, cMap="RdBu_r", nLevs=11, colorBar=False, extent=self.extent[:4])
    def plot_sensitivity(self, ax, extent):
        pg.show(self.mesh, self.normsens, cMap="RdGy_r", ax=ax, colorBar=False, nLevs=3, cMin=-1, cMax=1,  extent=self.extent[:4])

    def set_id_aruco(self, ids:dict):
        """
        key of the dictionary must be the aruco id and the value is the postion from 0 to 3.
        i.e. id = {12: 0, 20: 1, 13: 2, 4: 3}
        Args:
            ids:

        Returns:
        """
        if isinstance(ids, dict):
            self.id = ids
        else:
            print("Data type not accepted. Only accept dictionary as parameter")

    def set_aruco_electrodes(self, df):
        """
        Convert the aruco data frame into
        Args:
            df: pandas dataframe containing the markers
            id: dictionary indicating the id number and the corresponding order
        Returns:

        """
        df = df[['box_x', 'box_y']].copy()
        markers = numpy.zeros((4,2))
        try:
            for index in self.id.keys():
                markers[self.id[index], :] = df.loc[index]
            self.set_electrode_positions(markers)
        except:
            print("index "+str(index)+"  not found in aruco markers inside box, check your markers again")


    def update_resistivity(self, frame, extent, step):
        _ = self.create_mesh(frame, step, extent)
        _ = self.create_data_containerERT()
        _ = self.calculate_current_flow()

    def create_mesh(self, frame, step=7.5, extent=None):
        """
        create a grid mesh and populate the mesh with the frame data
        Args:
            frame:

        Returns:

        """
        if extent is None:
            y, x = frame.shape
        else:
            x = extent[1]
            y = extent[3]

        self.mesh_fine = mt.createGrid(numpy.arange(0, x, step=1),
                             numpy.arange(0, y, step=1))
        self.data_fine = mt.nodeDataToCellData(self.mesh_fine, frame.ravel())

        self.mesh = mt.createGrid(numpy.arange(0, x + step, step=step),
                             numpy.arange(0, y + step, step=step))
        self.data = pg.interpolate(srcMesh=self.mesh_fine, inVec=self.data_fine,
                              destPos=self.mesh.cellCenters()).array()
        return self.mesh, self.data

    def show_mesh(self):
        """
        Visualize the original and coarser resolution of the frame to use
        Returns:
            plot

        """
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=self._figsize)
        #plt.close()
        pg.show(self.mesh_fine, self.data_fine, ax=ax1, colorBar=True, extent=self.extent[:4], animated=True)
        pg.show(self.mesh, self.data, ax=ax2, colorBar=True, extent=self.extent[:4], animated=True)
        ax1.set_title("Original resolution")
        ax2.set_title("Coarser resolution")
        print("Original", self.mesh_fine)
        print("Coarse", self.mesh)
        print("Size reduction by %.2f%%" % float(100 - self.mesh.cellCount() / self.mesh_fine.cellCount() * 100))
        return fig

    def set_electrode_positions(self, markers=numpy.array(([20, 130],
                                                           [160, 20],
                                                           [50, 60],
                                                           [150, 120]))
                                ):
        """
        Use the arucos or a numpy array to position the electrodes
        Args:
            markers: numpy array of shape (4,2)

        Returns:

        """
        self.electrode = markers

    def create_data_containerERT(self, measurements = numpy.array([[0, 1, 2, 3],]), #Dipole-Dipole
                                 scheme_type = "abmn", verbose= False):
        """
        creates the scheme from the previous 4 aruco markers detected
        Args:
            measurements: Dipole-Dipole
            scheme_type: assign the type of electrode to the aruco

        Returns:

        """
        scheme = pg.DataContainerERT()
        scheme.setSensorPositions(self.electrode)
        for i, elec in enumerate(scheme_type):
            scheme[elec] = measurements[:, i]
        scheme["k"] = ert.createGeometricFactors(scheme, verbose=verbose)
        self.scheme = scheme
        return self.scheme

    def calculate_current_flow(self, time=False, verbose=False):
        """
        Perform the simulation based on the mesh, data and scheme
        Returns:
            RMatrix and RVector

        """
        if time:
            pg.tic()
        self.sim = ert.simulate(self.mesh, res=self.data, scheme=self.scheme, sr=False,
                           calcOnly=True, verbose=verbose, returnFields=True)
        if time:
            pg.toc("Current flow", box=True)
        self.pot = pg.utils.logDropTol(self.sim[0] - self.sim[1], 10)
        return self.sim, self.pot

    def show_streams(self):
        """
        Show the solution of the simulation and draw a stream plot perpendicular to the electric field
        Returns:

        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self._figsize, sharey=True)
        #plt.close()
        pg.show(self.mesh, self.data, ax=ax1, extent=self.extent[:4], animated=True)
        pg.show(self.mesh, self.pot, ax=ax2, nLevs=11, cMap="RdBu_r", extent=self.extent[:4], animated=True)

        for ax in ax1, ax2:
            ax.plot(self.electrode[0, 0], self.electrode[0, 1], "ro")  # Source
            ax.plot(self.electrode[1, 0], self.electrode[1, 1], "bo")  # Sink
            drawStreams(ax, self.mesh, -self.pot, color='Black')
        return fig

    def calculate_sensitivity(self, time=False):
        """
        Make a sensitivity analysis
        Returns:

        """
        if time:
            pg.tic()
        self.fop = ert.ERTModelling()
        self.fop.setData(self.scheme)
        self.fop.setMesh(self.mesh)
        self.fop.createJacobian(self.data)
        if time:
            pg.toc("Sensitivity calculation", box=True)
        sens = self.fop.jacobian()[0]  # first row = first measurement
        self.normsens = pg.utils.logDropTol(sens / self.mesh.cellSizes(), 5e-5)
        self.normsens /= numpy.max(self.normsens)
        return self.fop

    def show_sensitivity(self):
        """
        Show the sensitivity analysis
        Returns:

        """
        fig, ax = plt.subplots(figsize=self._figsize)
        #plt.close()
        pg.show(self.mesh, self.normsens, cMap="RdGy_r", orientation="vertical", ax=ax,
                label="Normalized\nsensitivity", nLevs=3, cMin=-1, cMax=1, extent=self.extent[:4], animated=True)
        for m in self.electrode:
            ax.plot(m[0], m[1], "ko")
        return fig

    def update_panel_mesh(self):
        self.figure_mesh.clf()
        self.axes_mesh.cla()
        self.figure_mesh.add_axes(self.axes_mesh)

        pg.show(self.mesh, self.data, ax=self.axes_mesh, colorBar=True, extent=self.extent[:4], animated=True)
        if self.electrode is not None:
            self.axes_mesh.plot(self.electrode[0, 0], self.electrode[0, 1], "ro")  # Source
            self.axes_mesh.plot(self.electrode[1, 0], self.electrode[1, 1], "bo")  # Sink
            if self.pot is not None:
                drawStreams(self.axes_mesh, self.mesh, -self.pot, color='Black')
        self.axes_mesh.set_title("Coarse Mesh")
        self.panel_figure_mesh.param.trigger("object")

    def update_panel_field(self):
        self.figure_field.clf()
        self.axes_field.cla()
        self.figure_field.add_axes(self.axes_field)
        pg.show(self.mesh, self.pot, ax=self.axes_field, nLevs=11, cMap="RdBu_r", extent=self.extent[:4], animated=True)
        if self.electrode is not None:
            self.axes_field.plot(self.electrode[0, 0], self.electrode[0, 1], "ro")  # Source
            self.axes_field.plot(self.electrode[1, 0], self.electrode[1, 1], "bo")  # Sink
            if self.pot is not None:
                drawStreams(self.axes_field, self.mesh, -self.pot, color='Black')
        self.axes_field.set_title("Current flow")
        self.panel_figure_field.param.trigger("object")

    def update_panel_sensitivity(self):
        self.figure_sensitivity.clf()
        self.axes_sensitivity.cla()
        self.figure_sensitivity.add_axes(self.axes_sensitivity)

        pg.show(self.mesh, self.normsens, cMap="RdGy_r", orientation="vertical", ax=self.axes_sensitivity,
                label="Normalized\nsensitivity", nLevs=3, cMin=-1, cMax=1, extent=self.extent[:4], animated=True)
        for m in self.electrode:
            self.axes_sensitivity.plot(m[0], m[1], "ko")
        self.axes_sensitivity.set_title("Sensitivity")
        self.panel_figure_sensitivity.param.trigger("object")

    def show_widgets(self):
        controller = self._create_widgets_controller()
        simulation = self._create_widgets_simulation()

        panel = pn.Row(pn.Column(simulation, controller),
                       pn.Column(self.panel_figure_mesh,
                                 self.panel_figure_field,
                                 self.panel_figure_sensitivity))
        return panel

    def _create_widgets_controller(self):
        self._widget_visualize = pn.widgets.RadioBoxGroup(name='Show the following mode in the sandbox:',
                                                          options=["mesh", "potential", "sensitivity"],
                                                          value=self.view,
                                                          inline=False)
        self._widget_visualize.param.watch(self._callback_visualize, 'value', onlychanged=False)

        self._widget_p_stream = pn.widgets.Checkbox(name='Show current stream lines', value=self.p_stream)
        self._widget_p_stream.param.watch(self._callback_p_stream, 'value', onlychanged=False)

        panel = pn.Column("### Controllers",
                          "<b>Visualization mode in the sandbox",
                          self._widget_visualize,
                          self._widget_p_stream
                          )
        return panel

    def _create_widgets_simulation(self):
        self._widget_step = pn.widgets.Spinner(name="Coarsen the mesh by", value=self.step, step=0.1)

        self._widget_create_mesh = pn.widgets.Button(name="Create mesh", button_type="success")
        self._widget_create_mesh.param.watch(self._callback_create_mesh, 'clicks', onlychanged=False)
        s = """<p>Original None </p><p>.</p><p>.</p><p>Coarser None</p><p>.</p><p>.</p><p>Size reduction by None</p><p>.</p>"""
        self._widget_markdown_mesh = pn.pane.Markdown(s)

        self._widget_a = pn.widgets.Spinner(name="Positive current electrode A", value=self._a_id, step=1)
        self._widget_b = pn.widgets.Spinner(name="Negative current electrode B", value=self._b_id, step=1)
        self._widget_m = pn.widgets.Spinner(name="Potential electrode M", value=self._m_id, step=1)
        self._widget_n = pn.widgets.Spinner(name="Potential electrode N", value=self._n_id, step=1)

        self._widget_set_electrodes = pn.widgets.Button(name="Set aruco electrodes", button_type="success")
        self._widget_set_electrodes.param.watch(self._callback_set_electrodes, 'clicks', onlychanged=False)

        s = "<p>Electrodes not set</p><p>.</p>"
        self._widget_markdown_electrodes = pn.pane.Markdown(s)

        self._widget_simulate = pn.widgets.Button(name="Simulate ert", button_type="success")
        self._widget_simulate.param.watch(self._callback_simulate, 'clicks', onlychanged=False)

        self._widget_sensitivity = pn.widgets.Button(name="Sensitivity analysis", button_type="success")
        self._widget_sensitivity.param.watch(self._callback_sensitivity, 'clicks', onlychanged=False)

        panel = pn.Column("###Simulation",
                          "<b>1) Create the mesh</b>",
                          self._widget_step,
                          self._widget_create_mesh,
                          self._widget_markdown_mesh,
                          "<b>2) Assign id aruco markers",
                          self._widget_a,
                          self._widget_b,
                          self._widget_m,
                          self._widget_n,
                          self._widget_set_electrodes,
                          self._widget_markdown_electrodes,
                          "<b>3) Calculate current-flow",
                          self._widget_simulate,
                          "<b>4) Calculate sensitivity",
                          self._widget_sensitivity
                          )
        return panel


    def _callback_visualize(self, event):
        self.view = event.new

    def _callback_p_stream(self, event):
        self.p_stream = event.new

    def _callback_create_mesh(self, event):
        self.lock.acquire()
        self.step = self._widget_step.value
        self.create_mesh(frame=self.frame, step=self.step)
        self.update_panel_mesh()
        s = """<p>Original %s </p> <p>Coarse %s </p> <p>Size reduction by %.2f%%</p>""" % \
            (str(self.mesh),
             str(self.mesh_fine),
             float(100 - self.mesh.cellCount() / self.mesh_fine.cellCount() * 100))
        self._widget_markdown_mesh.object = s
        self.lock.release()

    def _callback_set_electrodes(self, event):
        self.lock.acquire()
        a = self._widget_a.value
        b = self._widget_b.value
        m = self._widget_m.value
        n = self._widget_n.value
        self.set_id_aruco({a: 0,
                           b: 1,
                           m: 2,
                           n: 3})
        self.set_aruco_electrodes(self._df_markers)
        if self.electrode is not None:
            self._widget_markdown_electrodes.object = "Electrodes: " + str(self.id)
        else:
            self._widget_markdown_electrodes.object = "Error, check ids"
            self.lock.release()
            return
        self.create_data_containerERT()
        self.update_panel_mesh()
        self.lock.release()

    def _callback_simulate(self, event):
        self.lock.acquire()
        self.calculate_current_flow()
        self.update_panel_mesh()
        self.update_panel_field()
        self.lock.release()

    def _callback_sensitivity(self, event):
        self.lock.acquire()
        self.calculate_sensitivity()
        self.update_panel_sensitivity()
        self.lock.release()