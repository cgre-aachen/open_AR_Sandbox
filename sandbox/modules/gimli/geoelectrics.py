import matplotlib.pyplot as plt
import numpy
import pygimli as pg
import pygimli.meshtools as mt
import pygimli.physics.ert as ert
from pygimli.viewer.mpl import drawStreams
print(pg.__version__)

from ..template import ModuleTemplate

class GeoelectricsModule(ModuleTemplate):
    """
    Module to show the potential fields of a geoelectric (DC resistivity) forward modelling and sesitivity analysis
    check "https://www.pygimli.org/_examples_auto/3_dc_and_ip/plot_04_ert_2_5d_potential.html#sphx-glr-examples-auto-3-dc-and-ip-plot-04-ert-2-5d-potential-py" for more information
    """
    def __init__(self, *args, extent: list = None, **kwargs):
        # call parents' class init, use greyscale colormap as standard and extreme color labeling
        if extent is not None:
            self.vmin = extent[4]
            self.vmax = extent[5]

        self.mesh_fine = None
        self.mesh = None
        self.data_fine = None
        self.data = None
        self.scheme = None
        self.sim = None
        self.pot = None
        self.fop = None
        self.normsens = None

        self.step = 7.5
        self.sensitivity = False
        self.view = "mesh"

    def update(self, sb_params: dict):
        frame = sb_params.get('frame') * 1000 + 50
        extent = sb_params.get('extent')
        ax = sb_params.get('ax')
        markers = sb_params.get('marker')
        cmap = sb_params.get('cmap')
        if len(markers.loc[markers.is_inside_box])==4 and self.id is not None:
            self.set_aruco_electrodes(markers)
            self.update_resistivity(frame, extent, self.step)
            if self.sensitivity:
                self.calculate_sensitivity()

        ax = self.plot(ax, frame, cmap)

        return sb_params

    def plot(self, ax, frame, cmap):
        ax.cla()
        if self.view == "mesh":
            self.plot_mesh(ax, frame, cmap)
        elif self.view == "potential" and self.pot is not None:
            self.plot_potential(ax)
        elif self.view == "sensitivity" and self.normsens is not None:
            self.plot_sensitivity(ax)
        if self.stream and self.pot is not None:
            self.plot_stream_lines(ax)
        return ax

    def plot_stream_lines(self, ax): drawStreams(ax, self.mesh, -self.pot, color='Black')
    def plot_mesh(self, ax, frame, cmap):
        ax.imshow(frame, origin="lower", cmap=cmap)
    def plot_potential(self, ax):
        pg.show(self.mesh, self.pot, ax=ax, cmap="RdBu_r", nLevs=11, colorBar=False, )
    def plot_sensitivity(self, ax):
        pg.show(self.mesh, self.normsens, cmap="RdGy_r", ax=ax, colorBar=False, nLevs=3, cMin=-1, cMax=1)

    def set_id_aruco(self, id:dict):
        """
        key of the dictionary must be the aruco id and the value is the postion from 0 to 3.
        i.e. id = {12: 0, 20: 1, 13: 2, 4: 3}
        Args:
            id:

        Returns:
        """
        if isinstance(id, dict):
            self.id = id
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
        #df[id]
        markers = numpy.zeros((4,2))
        for index in self.id.keys():
            markers[self.id[index], :] = df.iloc[index].loc[df.is_inside_box, ('box_x', 'box_y')].values
        self.set_electrode_positions(markers)


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
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        pg.show(self.mesh_fine, self.data_fine, ax=ax1, colorBar=False)
        pg.show(self.mesh, self.data, ax=ax2, colorBar=False)
        ax1.set_title("Original resolution")
        ax2.set_title("Coarser resolution")
        print("Original", self.mesh_fine)
        print("Coarse", self.mesh)
        print("Size reduction by %.2f%%" % float(100 - self.mesh.cellCount() / self.mesh_fine.cellCount() * 100))
        fig.show()

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

    def create_data_containerERT(self, measurements = numpy.array(([0, 1, 2, 3],)), #Dipole-Dipole
                                 scheme_type = "abmn"):
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
        scheme["k"] = ert.createGeometricFactors(scheme)
        self.scheme = scheme
        return self.scheme

    def calculate_current_flow(self):
        """
        Perform the simulation based on the mesh, data and scheme
        Returns:

        """
        pg.tic()
        self.sim = ert.simulate(self.mesh, res=self.data, scheme=self.scheme, sr=False,
                           calcOnly=True, verbose=True, returnFields=True)
        pg.toc("Current flow", box=True)
        self.pot = pg.utils.logDropTol(self.sim[0] - self.sim[1], 10)
        return self.sim, self.pot

    def show_streams(self):
        """
        Show the solution of the simulation and draw a stream plot perpendicular to the electric field
        Returns:

        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        pg.show(self.mesh, self.data, ax=ax1)
        pg.show(self.mesh, self.pot, ax=ax2, nLevs=11, cmap="RdBu_r")

        for ax in ax1, ax2:
            ax.plot(self.electrode[0, 0], self.electrode[0, 1], "ro")  # Source
            ax.plot(self.electrode[1, 0], self.electrode[1, 1], "bo")  # Sink
            drawStreams(ax, self.mesh, -self.pot, color='Black')
        fig.show()

    def calculate_sensitivity(self):
        """
        Make a sensitivity analysis
        Returns:

        """
        pg.tic()
        self.fop = ert.ERTModelling()
        self.fop.setData(self.scheme)
        self.fop.setMesh(self.mesh)
        self.fop.createJacobian(self.data)
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
        fig, ax = plt.subplots()
        pg.show(self.mesh, self.normsens, cMap="RdGy_r", orientation="vertical", ax=ax,
                label="Normalized\nsensitivity", nLevs=3, cMin=-1, cMax=1)
        for m in self.electrode:
            ax.plot(m[0], m[1], "ko")
        fig.show()

    def show_widgets(self):
        pass
