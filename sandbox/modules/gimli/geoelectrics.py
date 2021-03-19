import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import panel as pn
import numpy
import pygimli as pg
import pygimli.meshtools as mt
import pygimli.physics.ert as ert
from pygimli.viewer.mpl.meshview import drawStreamLines, drawStreams
from ..template import ModuleTemplate
from sandbox import set_logger
logger = set_logger(__name__)

import logging

pg.setLogLevel(logging.WARNING)  # to reduce logging output


class GeoelectricsModule(ModuleTemplate):
    """
    Module to show the potential fields of a geoelectric (DC resistivity) forward modelling and sesitivity analysis
    check "https://www.pygimli.org/_examples_auto/3_dc_and_ip/plot_04_ert_2_5d_potential.html#sphx-glr-examples-auto-3-dc-and-ip-plot-04-ert-2-5d-potential-py"
    for more information
    """

    def __init__(self, *args, extent: list = None, **kwargs):
        if extent is not None:
            self.extent = extent
        self.vmin = 5.  # ohm*m
        self.vmax = 1e5  # ohm*m
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
        self.sensitivity_real_time = False
        self.view = "mesh"  # "potential"  "sensitivity""
        self.id = None
        self.real_time = False
        self.p_stream = False
        self.p_quiver = False
        self.frame = None
        self.df_markers = []
        self._figsize = (15, 8)
        self._recalculate_mesh = False

        # Widgets
        self._a_id = 0
        self._b_id = 3
        self._n_id = 1
        self._m_id = 2
        self.figure_mesh = Figure()
        self.panel_figure_mesh = pn.pane.Matplotlib(self.figure_mesh, tight=True)
        plt.close(self.figure_mesh)

        self.figure_field = Figure()
        self.panel_figure_field = pn.pane.Matplotlib(self.figure_field, tight=True)
        plt.close(self.figure_field)

        self.figure_sensitivity = Figure()
        self.panel_figure_sensitivity = pn.pane.Matplotlib(self.figure_sensitivity, tight=True)
        plt.close(self.figure_sensitivity)

        logger.info("GeoelectricsModule loaded successfully")

    def update(self, sb_params: dict):
        frame = sb_params.get('frame')
        extent = sb_params.get('extent')
        frame, extent = self.scale_data(frame, extent)
        ax = sb_params.get('ax')
        markers = sb_params.get('marker')
        self.lock = sb_params.get("lock_thread")
        if len(markers) > 0:
            self.df_markers = markers.loc[markers.is_inside_box]
            if len(self.df_markers) == 4 and self.id is not None:
                self.set_aruco_electrodes(self.df_markers)
                if self.real_time:
                    self.update_resistivity(frame, self.extent, self.step)
                    if self.sensitivity_real_time:
                        self.calculate_sensitivity()

        ax, activate_c = self.plot(ax)
        sb_params["frame"] = frame
        sb_params["extent"] = extent
        sb_params["active_contours"] = activate_c
        return sb_params

    def plot(self, ax):
        self.delete_image(ax)
        activate_c = True
        if self.p_stream and self.pot is not None:
            self.plot_stream_lines(ax)
            activate_c = False
        if self.p_quiver and self.pot is not None:
            self.plot_quiver(ax)
            activate_c = False
        if self.view == "potential" and self.pot is not None:
            self.plot_potential(ax)
            activate_c = False
        elif self.view == "sensitivity" and self.normsens is not None:
            self.plot_sensitivity(ax)
            activate_c = False
        return ax, activate_c

    def scale_data(self, frame, extent, vmax: float = None, vmin: float = None):
        """
        Find a better way to do the scaling of the data
        Args:
            frame: sandbox frame
            extent: sandbox extent
            vmin: minimum value of ohm*m
            vmax: maximum value of ohm*m
        Returns:
            modifyied sandbox frame and extent(vmin and vmax)
        """
        if vmin is None:
            vmin = self.vmin
        if vmax is None:
            vmax = self.vmax
        frame = frame * 1000 + 50
        extent[-1] = extent[-1] * 1000 + 50  # max_height
        extent[-2] = extent[-2] * 1000 + 50  # min_height
        self.frame = frame
        self.extent = extent
        return self.frame, self.extent

    def scale_linear(self, frame, extent, vmin: float = None, vmax: float = None):
        """
        DOES NOT WORK WITH THIS SCALING!!! Neds to be more extreme
        Modify the frame to get some realistic Ohm*m values
        # TODO: Normalize logaritmically? to have more extreme values
        Args:
            frame: sandbox frame
            extent: sandbox extent
            vmin: minimum value of ohm*m
            vmax: maximum value of ohm*m
        Returns:
            modifyied sandbox frame and extent(vmin and vmax)
        """
        if vmin is None:
            vmin = self.vmin
        if vmax is None:
            vmax = self.vmax

        frame = frame * (vmax - vmin) / (extent[-1] - extent[-2])
        frame = frame + vmin

        extent[-1] = vmax  # max_height
        extent[-2] = vmin  # min_height

        self.frame = frame
        self.extent = extent
        return self.frame, self.extent

    def delete_image(self, ax):
        # [coll.remove() for coll in reversed(ax.collections) if isinstance(coll, matplotlib.collections.PathCollection )]
        ax.cla()  # TODO: find a better way to do this
        ax.set_axis_off()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    def plot_stream_lines(self, ax):
        drawStreamLines(ax, self.mesh, self.pot, color='Black', nx=100, ny=100, linewidth=1.0, zorder=50)

    def plot_quiver(self, ax):
        drawStreams(ax, self.mesh, pg.solver.grad(self.mesh, -self.pot), color='Black', quiver=True, zorder=49)

    def plot_potential(self, ax):
        pg.show(self.mesh, self.pot, ax=ax, cMap="RdBu_r", nLevs=11, colorBar=False, extent=self.extent[:4], zorder=20)

    def plot_sensitivity(self, ax):
        pg.show(self.mesh, self.normsens, cMap="RdGy_r", ax=ax, colorBar=False, nLevs=3, cMin=-1, cMax=1,
                extent=self.extent[:4], zorder=30)

    def set_id_aruco(self, ids: dict):
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
            logger.warning("Data type not accepted. Only accept dictionary as parameter")

    def set_aruco_electrodes(self, df):
        """
        Convert the aruco data frame into
        Args:
            df: pandas dataframe containing the markers
            id: dictionary indicating the id number and the corresponding order
        Returns:

        """
        df = df[['box_x', 'box_y']].copy()
        markers = numpy.zeros((4, 2))
        try:
            for index in self.id.keys():
                markers[self.id[index], :] = df.loc[index]
            self.set_electrode_positions(markers)
        except:
            logger.warning("index " + str(index) + "  not found in aruco markers inside box, check your markers again")

    def update_resistivity(self, frame, extent, step):
        if self.mesh is None:
            _ = self.create_mesh(frame, step, extent)
        if self._recalculate_mesh:
            _ = self.create_mesh(frame, step, extent)
        _ = self.create_data_containerERT()
        _ = self.calculate_current_flow()

    def create_mesh(self, frame, step=7.5, extent=None):
        """
        create a grid mesh and populate the mesh with the frame data
        Args:
            frame:
            step:
            extent:
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
        # plt.close(fig)
        pg.show(self.mesh_fine, self.data_fine, ax=ax1, colorBar=True, label=r"Resistivity ($\Omega m$)",
                extent=self.extent[:4])
        pg.show(self.mesh, self.data, ax=ax2, colorBar=True, label=r"Resistivity ($\Omega m$)",
                extent=self.extent[:4])
        ax1.set_title("Original resolution")
        ax2.set_title("Coarser resolution")
        logger.info("Original %s" % self.mesh_fine)
        logger.info("Coarse %s" % self.mesh)
        logger.info("Size reduction by %.2f%%" % float(100 - self.mesh.cellCount() / self.mesh_fine.cellCount() * 100))
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

    def create_data_containerERT(self, measurements=numpy.array([[0, 1, 2, 3], ]),  # Dipole-Dipole
                                 scheme_type="abmn", verbose=False):
        """
        creates the scheme from the previous 4 aruco markers detected
        Args:
            measurements: Dipole-Dipole
            scheme_type: assign the type of electrode to the aruco
            verbose:

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

    def show_streams(self, quiver=False):
        """
        Show the solution of the simulation and draw a stream plot perpendicular to the electric field
        Returns:

        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self._figsize, sharey=True)
        #plt.close(fig)
        pg.show(self.mesh, self.data, ax=ax1, label="Resistivity ($\Omega m$)", extent=self.extent[:4])
        pg.show(self.mesh, self.pot, ax=ax2, nLevs=11, cMap="RdBu_r",
                label="Potential ($u$)", extent=self.extent[:4])

        for ax in ax1, ax2:
            ax.plot(self.electrode[0, 0], self.electrode[0, 1], "ro")  # Source
            ax.plot(self.electrode[1, 0], self.electrode[1, 1], "bo")  # Sink
            drawStreamLines(ax, self.mesh, self.pot, color='Black', nx=100, ny=100, linewidth=1.0)
            if quiver:
                drawStreams(ax, self.mesh, pg.solver.grad(self.mesh, -self.pot), color='Black', quiver=True)
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
        #plt.close(fig)
        pg.show(self.mesh, self.normsens, cMap="RdGy_r", orientation="vertical", ax=ax,
                label="Normalized\nsensitivity", nLevs=3, cMin=-1, cMax=1, extent=self.extent[:4])
        for m in self.electrode:
            ax.plot(m[0], m[1], "ko")
        return fig

    def update_panel_mesh(self):
        fig, ax = plt.subplots()
        plt.close(fig)
        pg.show(self.mesh, self.data, ax=ax, colorBar=True, extent=self.extent[:4],
                orientation="vertical", label=r"Resistivity ($\Omega m$)")
        if self.electrode is not None:
            ax.plot(self.electrode[0, 0], self.electrode[0, 1], "ro")  # Source
            ax.plot(self.electrode[1, 0], self.electrode[1, 1], "bo")  # Sink
            if self.pot is not None:
                drawStreamLines(ax, self.mesh, self.pot, color='Black', nx=100, ny=100, linewidth=1.0)
        ax.set_title("Coarse Mesh")
        self.panel_figure_mesh.object = fig
        self.panel_figure_mesh.param.trigger("object")

    def update_panel_field(self):
        fig, ax = plt.subplots()
        plt.close(fig)
        pg.show(self.mesh, self.pot, ax=ax, nLevs=11, cMap="RdBu_r", extent=self.extent[:4],
                label="Potential ($u$)", orientation="vertical")
        if self.electrode is not None:
            ax.plot(self.electrode[0, 0], self.electrode[0, 1], "ro")  # Source
            ax.plot(self.electrode[1, 0], self.electrode[1, 1], "bo")  # Sink
            if self.pot is not None:
                drawStreamLines(ax, self.mesh, self.pot, color='Black', nx=100, ny=100, linewidth=1.0)
        ax.set_title("Current Flow")
        self.panel_figure_field.object = fig
        self.panel_figure_field.param.trigger("object")

    def update_panel_sensitivity(self):
        fig, ax = plt.subplots()
        plt.close(fig)
        pg.show(self.mesh, self.normsens, cMap="RdGy_r", orientation="vertical", ax=ax,
                label="Normalized\nsensitivity", nLevs=3, cMin=-1, cMax=1, extent=self.extent[:4])
        for m in self.electrode:
            ax.plot(m[0], m[1], "ko")
        ax.set_title("Sensitivity")
        self.panel_figure_sensitivity.object = fig
        self.panel_figure_sensitivity.param.trigger("object")

    def show_widgets(self):
        controller, real = self.widgets_controller()
        simulation = self.widgets_simulation()

        self._widget_progress = pn.widgets.Progress(name="Progress", active=True)
        self._widget_markdown_progress = pn.pane.Markdown("<p>Waiting</p><p>.</p>")

        panel = pn.Row(simulation,
                       pn.Column(self.panel_figure_mesh,
                                 self.panel_figure_field,
                                 self.panel_figure_sensitivity,
                                 self._widget_progress,
                                 self._widget_markdown_progress),
                       pn.Column(controller, real))
        return panel

    def widgets_controller(self):
        self._widget_visualize = pn.widgets.RadioBoxGroup(name='Show the following mode in the sandbox:',
                                                          options=["mesh", "potential", "sensitivity"],
                                                          value=self.view,
                                                          inline=False)
        self._widget_visualize.param.watch(self._callback_visualize, 'value', onlychanged=False)

        self._widget_p_stream = pn.widgets.Checkbox(name='Show current stream lines', value=self.p_stream)
        self._widget_p_stream.param.watch(self._callback_p_stream, 'value', onlychanged=False)

        self._widget_p_quiver = pn.widgets.Checkbox(name='Show vector field', value=self.p_quiver)
        self._widget_p_quiver.param.watch(self._callback_p_quiver, 'value', onlychanged=False)

        panel = pn.Column("###<b>Controllers</b>",
                          "<b>Visualization mode in the sandbox",
                          self._widget_visualize,
                          self._widget_p_stream,
                          self._widget_p_quiver
                          )
        self._widget_real_time = pn.widgets.Checkbox(name='Activate real time calculation',
                                                     value=self.real_time)
        self._widget_real_time.param.watch(self._callback_real_time, 'value', onlychanged=False)

        self._widget_recalculate_mesh = pn.widgets.Checkbox(name='Create mesh for every frame',
                                                            value=self._recalculate_mesh)
        self._widget_recalculate_mesh.param.watch(self._callback_recalculate, 'value', onlychanged=False)

        self._widget_recalculate_sensitivity = pn.widgets.Checkbox(name='Calculate sensitivity for every frame',
                                                                   value=self.sensitivity_real_time)
        self._widget_recalculate_sensitivity.param.watch(self._callback_recalculate_sensitivity, 'value',
                                                         onlychanged=False)
        self._widget_update_plots = pn.widgets.Button(name="Update all plots", button_type="success")
        self._widget_update_plots.param.watch(self._callback_update_plots, 'clicks', onlychanged=False)

        real = pn.WidgetBox("###<b>Update real time</b>",
                            self._widget_real_time,
                            self._widget_recalculate_mesh,
                            self._widget_recalculate_sensitivity,
                            "Remember that all 4 arucos must be detected and assigned to their respective electrodes. "
                            "Otherwise the real time calculation is stopped until it "
                            "can detect all electrodes",
                            self._widget_update_plots)
        return panel, real

    def widgets_simulation(self):
        self._widget_step = pn.widgets.Spinner(name="Coarsen the mesh by", value=self.step, step=0.1)
        self._widget_vmin = pn.widgets.Spinner(name="Minimum resistivity (ohm*m)", value=self.vmin, step=1)
        self._widget_vmax = pn.widgets.Spinner(name="Maximum resitivity (ohm*m)", value=self.vmax, step=1)

        self._widget_create_mesh = pn.widgets.Button(name="Create mesh", button_type="success")
        self._widget_create_mesh.param.watch(self._callback_create_mesh, 'clicks', onlychanged=False)
        s = "<p>Original None </p><p>.</p>" + \
            "<p>.</p> <p>Coarser None</p><p>.</p><p>.</p><p>Size reduction by None</p><p>.</p>"
        self._widget_markdown_mesh = pn.pane.Markdown(s)

        self._widget_a = pn.widgets.Spinner(name="Positive current electrode A", value=self._a_id, step=1)
        self._widget_b = pn.widgets.Spinner(name="Negative current electrode B", value=self._b_id, step=1)
        self._widget_m = pn.widgets.Spinner(name="Potential electrode M", value=self._m_id, step=1)
        self._widget_n = pn.widgets.Spinner(name="Potential electrode N", value=self._n_id, step=1)

        self._widget_set_electrodes = pn.widgets.Button(name="Set aruco electrodes", button_type="success")
        self._widget_set_electrodes.param.watch(self._callback_set_electrodes, 'clicks', onlychanged=False)

        s = "<p>Electrodes not set</p><p>.</p><p>.</p>"
        self._widget_markdown_electrodes = pn.pane.Markdown(s)

        self._widget_simulate = pn.widgets.Button(name="Simulate ert", button_type="success")
        self._widget_simulate.param.watch(self._callback_simulate, 'clicks', onlychanged=False)

        self._widget_sensitivity = pn.widgets.Button(name="Sensitivity analysis", button_type="success")
        self._widget_sensitivity.param.watch(self._callback_sensitivity, 'clicks', onlychanged=False)

        panel = pn.Column("###<b>Simulation</b>",
                          "<b>1) Create the mesh</b>",
                          self._widget_vmax,
                          self._widget_vmin,
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

    def _callback_update_plots(self, event):
        self.lock.acquire()
        self._widget_markdown_progress.object = "Updating mesh plot..."
        logger.info("Updating mesh plot...")
        self._widget_progress.value = 0
        self.update_panel_mesh()
        self._widget_markdown_progress.object = "Updating field plot..."
        logger.info("Updating field plot...")
        self._widget_progress.value = 40
        self.update_panel_field()
        self._widget_markdown_progress.object = "Updating sensitivity plot..."
        logger.info("Updating sensitivity plot...")
        self._widget_progress.value = 80
        self.update_panel_sensitivity()
        self._widget_markdown_progress.object = "Done"
        logger.info("Done")
        self._widget_progress.value = 100
        self.lock.release()

    def _callback_recalculate_sensitivity(self, event):
        self.sensitivity_real_time = event.new

    def _callback_recalculate(self, event):
        self._recalculate_mesh = event.new

    def _callback_real_time(self, event):
        self.real_time = event.new

    def _callback_visualize(self, event):
        self.view = event.new

    def _callback_p_stream(self, event):
        self.p_stream = event.new

    def _callback_p_quiver(self, event):
        self.p_quiver = event.new

    def _callback_create_mesh(self, event):
        self.lock.acquire()
        self._widget_markdown_progress.object = "Creating mesh..."
        logger.info("Creating mesh...")
        self._widget_progress.value = 0
        self.vmin = self._widget_vmin.value
        self.vmax = self._widget_vmax.value
        self.step = self._widget_step.value
        self.create_mesh(frame=self.frame, step=self.step)
        self._widget_progress.value = 50
        self._widget_markdown_progress.object = "Updating plot..."
        logger.info("Updating plot...")
        self.update_panel_mesh()
        self._widget_progress.value = 100
        s = """<p>Original %s </p> <p>Coarse %s </p> <p>Size reduction by %.2f%%</p>""" % \
            (str(self.mesh),
             str(self.mesh_fine),
             float(100 - self.mesh.cellCount() / self.mesh_fine.cellCount() * 100))
        self._widget_markdown_mesh.object = s
        logger.info(s)
        self._widget_markdown_progress.object = "Mesh ready"
        logger.info("Mesh ready")
        self.lock.release()

    def _callback_set_electrodes(self, event):
        self.lock.acquire()
        self._widget_markdown_progress.object = "Setting electrodes..."
        logger.info("Setting electrodes...")
        self._widget_progress.value = 0
        a = self._widget_a.value
        b = self._widget_b.value
        m = self._widget_m.value
        n = self._widget_n.value
        self.set_id_aruco({a: 0,
                           b: 1,
                           m: 2,
                           n: 3})
        self.set_aruco_electrodes(self.df_markers)
        self._widget_progress.value = 20
        if self.electrode is not None:
            self._widget_markdown_electrodes.object = "Ready electrodes: " + str(self.id)
            logger.info("Ready electrodes: " + str(self.id))
            self._widget_progress.value = 50
        else:
            self._widget_markdown_electrodes.object = "Error, check ids"
            logger.info("Error, check ids")
            self._widget_markdown_progress.object = "Error setting electrodes..."
            logger.info("Error setting electrodes...")
            self.lock.release()
            return
        self.create_data_containerERT()
        self._widget_progress.value = 80
        self._widget_markdown_progress.object = "Updating plot..."
        logger.info("Updating plot...")
        self.update_panel_mesh()
        self._widget_progress.value = 100
        self._widget_markdown_progress.object = "Electrodes ready"
        logger.info("Electrodes ready")
        self.lock.release()

    def _callback_simulate(self, event):
        self.lock.acquire()
        self._widget_progress.value = 0
        self._widget_markdown_progress.object = "Simulating..."
        logger.info("Simulating...")
        self.calculate_current_flow()
        self._widget_progress.value = 80
        self._widget_markdown_progress.object = "Updating plot 1..."
        logger.info("Updating plot 1...")
        self.update_panel_mesh()
        self._widget_progress.value = 90
        self._widget_markdown_progress.object = "Updating plot 2..."
        logger.info("Updating plot 2...")
        self.update_panel_field()
        self._widget_progress.value = 100
        self._widget_markdown_progress.object = "Simulation successful"
        logger.info("Simulation successful")
        self.lock.release()

    def _callback_sensitivity(self, event):
        self.lock.acquire()
        self._widget_progress.value = 0
        self._widget_markdown_progress.object = "Calculating sensitivity..."
        logger.info("Calculating sensitivity...")
        self.calculate_sensitivity()
        self._widget_progress.value = 80
        self._widget_markdown_progress.object = "Updating plot..."
        logger.info("Updating plot...")
        self.update_panel_sensitivity()
        self._widget_progress.value = 100
        self._widget_markdown_progress.object = "Sensitivity successful"
        logger.info("Sensitivity successful")
        self.lock.release()
