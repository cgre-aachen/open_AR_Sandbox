import sys, os

from .template import ModuleTemplate
from sandbox import _package_dir

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from matplotlib import cm
import traceback


class SeismicModule(ModuleTemplate):
    """ Follow the link to see the code that inspired the module
    https://nbviewer.jupyter.org/github/devitocodes/devito/blob/master/examples/seismic/tutorials/01_modelling.ipynb"""
    def __init__(self, devito_dir:str =_package_dir+'/../../devito/', **kwargs):
        """Importing packages and setting correctly the path to the devito package, this beacuse we need the examples"""
        try:
            sys.path.append(os.path.abspath(devito_dir) + '/examples/seismic/')
            from model import Model
            self.Model = Model
            from plotting import plot_velocity, plot_shotrecord
            self.plot_velocity = plot_velocity
            self.plot_shotrecord = plot_shotrecord
            from source import RickerSource, TimeAxis, Receiver
            self.RickerSource = RickerSource
            self.TimeAxis = TimeAxis
            self.Receiver = Receiver
            from devito import TimeFunction, Eq, solve, Operator
            self.TimeFunction = TimeFunction
            self.Eq=Eq
            self.solve = solve
            self.Operator = Operator
            print('Importing packages succesfull ')
        except Exception:
            traceback.print_exc()
        self.vp = None #Normalized or smoothed or None topography to be the velocity model
        self.model: Model = None #the model
        self.time_range = None #the time of the simulation
        self.src = None #the ricker source
        self.u = None #wavefield time function
        self.pde = None #PDE to solve. Wave equation with corresponding discretizations
        self.stencil= None # a time marching updating equation known as a stencil using customized SymPy functions
        self.src_term = [] #all sources together
        self.rec_term = [] #all receivers
        self.op = None #aAfter constructing all the necessary expressions for updating the wavefield, injecting the source term and interpolating onto the receiver points, we can now create the Devito operator
        self.n_frames = 50

    def update(self, sb_params: dict):
        frame = sb_params.get('frame')
        ax = sb_params.get('ax')
        marker = sb_params.get('marker')

    def plot(self):
        pass

    def scale_linear(self, topo: np.ndarray, high: float, low:float):
        """
        scale the frame data according to the highest and lowest value desired
        Args:
            topo: topo frame
            high: max value
            low: min value
        Returns:
            Normalized frame
        """
        mins = np.amin(topo)
        maxs = np.amax(topo)
        rng = maxs - mins
        return high - (((high - low) * (maxs - topo)) / rng)

    def smooth_topo(self, topo: np.ndarray, sigma_x:int, sigma_y:int):
        """
        Smoothing  the topography...
        Args:
            topo: topo frame
            sigma_x: smooth in x direction
            sigma_y: smooth in y direction
        Returns:
            Smothed array
        """
        sigma = [sigma_y, sigma_x]
        return ndimage.filters.gaussian_filter(topo, sigma, mode='nearest')

    def create_velocity_model(self, topo: np.ndarray, norm: bool=True, vmax:float=5.0,
                              vmin:float=2.0, smooth:bool=True, sigma_x:int=2, sigma_y:int=2,
                              spacing:tuple = (10, 10), origin=(0,0),
                              nbl:int=10, show_velocity: bool=True):
        """
        takes the topography and creates a velocity model from the topography
        TODO: introduce circles???
        Args:
            topo: topo array
            norm: to normalize the topo according to vmax and vmin
            vmax: maximum velocity for normalization
            vmin: minimum velocity for normalization
            smooth: to apply a gaussian filter to the topo
            sigma_x: smooth in x direction
            sigma_y: smooth in y_direction
            spacing: Grid spacing in m
            origin: What is the location of the top left corner. This is necessary to define the absolute location of the source and receivers
            nbl: size of the absorbing layer
            show_velocity: plot of the velocity model
        Returns:
            model

        """
        if topo is None:
            vp = np.empty((101, 101), dtype=np.float32)
            vp[:, :51] = 1.5
            vp[:, 51:] = 2.5
            topo=vp
        topo = topo.astype(np.float32)
        if norm:
            topo = self.scale_linear(topo, vmax, vmin)
        if smooth:
            topo = self.smooth_topo(topo, sigma_x, sigma_y)
        self.vp = topo
        self.model = self.Model(vp=topo, origin=origin, shape=topo.shape, spacing=spacing,
              space_order=2, nbl=nbl, bcs="damp")
        if show_velocity:
            self.plot_velocity(self.model)
        return self.model

    def create_time_axis(self, t0:int=0, tn:int=1000):
        """
        Time duration of our model. This takes the start, and end with the timestepsize provided by the model
        Args:
            t0:  Simulation starts a t=0
            tn:  Simulation last 1 second (1000 ms)
        Returns:
            Time_range
        """
        dt = self.model.critical_dt
        self.time_range = self.TimeAxis(start=t0, stop=tn, step=dt)
        return self.time_range

    @property
    def src_coords(self):
        """gives the source coordinates of the richter waveletet in the middle of the model"""
        #src_coords = np.multiply(circles[0], [dx, dy])
        return np.multiply([int(self.model.domain_size[0]/2),int(self.model.domain_size[1]/2/2)],
                                 [10, 10])

    def create_source(self, name:str = 'src', f0: float=0.025, source_coordinates:tuple=None, show_wavelet:bool=True, show_model:bool=True):
        """
        RickerSource positioned at a depth of "depth_source"
        Args:
            name: assign a name to the source, so we can distinguish if more sources
            f0: # Source peak frequency is 25Hz (0.025 kHz)
            source_coordinates: position (x, y) for the source. Scale is in meters
            show_wavelet: Plot the time signature to see the wavelet
            show_model: plot the velocity model and the location of the source
        Returns:
        """
        if source_coordinates is None:
            source_coordinates = self.src_coords

        self.src = self.RickerSource(name=name, grid=self.model.grid, f0=f0,
                                npoint=1, time_range=self.time_range, coordinates=source_coordinates)
        ## First, position source centrally in all dimensions, then set depth
        #src.coordinates.data[0, :] = np.array(model.domain_size) * .5
        #src.coordinates.data[0, -1] = 20.  # Depth is 20m
        if show_wavelet: # We can plot the time signature to see the wavelet
            self.src.show()
        if show_model:
            self.plot_velocity(self.model, source=self.src.coordinates.data)
        return self.src

    def create_time_function(self):
        """
        second order time discretization: This derivative is represented in Devito by u.dt2 where u is a TimeFunction object.
        Spatial discretization: This derivative is represented in Devito by u.laplace where u is a TimeFunction object.
        With space and time discretization defined, we can fully discretize the wave-equation with the combination of time and space discretizations
        Returns:

        """
        # Define the wavefield with the size of the model and the time dimension
        self.u = self.TimeFunction(name="u", grid=self.model.grid,
                              time_order=2, space_order=2)
        # We can now write the PDE
        self.pde = self.model.m * self.u.dt2 - self.u.laplace + self.model.damp * self.u.dt
        return self.pde

    def solve_PDE(self):
        """This discrete PDE can be solved in a time-marching way updating u(t+dt) from the previous time step"""
        self.stencil = self.Eq(self.u.forward, self.solve(self.pde, self.u.forward))
        return self.stencil

    def inject_source(self, source):
        """
        For every RickerSource we need to add it to the numerical scheme in order to solve the homogenous
        wave equation, in order to implement the measurement operator and interpolator operator.
        Args:
            source: from the self.create_source()
        Returns:
        """
        src_term = source.inject(field=self.u.forward, expr=source * self.model.critical_dt ** 2 / self.model.m)
        # TODO: offset=model.nbl))
        if self.src_term == []:
            self.src_term = src_term
        else:
            self.src_term += src_term
        return self.src_term

    def create_receivers(self, name:str = 'rec', n_receivers:int=50, depth_receivers:int=20, show_receivers:bool=True):
        """
        Interpolate the values of the receivers horizontaly at a depth
        Args:
            name: name to the receivers
            n_receivers: amount of points/receivers
            coordinates_origin: to plot the receivers along the x axis of the coordinate
        Returns:
        """
        x_locs = np.linspace(0, self.model.shape[0]*self.model.spacing[0], n_receivers)
        rec_coords = [(x, depth_receivers) for x in x_locs]
        self.rec = self.Receiver(name=name, npoint=n_receivers, time_range=self.time_range,
                        grid=self.model.grid, coordinates=rec_coords)
        if show_receivers:
            self.plot_velocity(self.model, receiver=self.rec.coordinates.data[::4, :])
        return self.rec

    def interpolate_receiver(self):
        """obtain the receivers information"""
        self.rec_term = self.rec.interpolate(expr=self.u.forward)
        return self.rec_term

    def operator_and_solve(self):
        """
        After constructing all the necessary expressions for updating the wavefield, injecting the source term and
        interpolating onto the receiver points, we can now create the Devito operator that will generate the C code at runtime.
        Returns:

        """
        self.op = self.Operator([self.stencil] + self.src_term + self.rec_term)#, subs=self.model.spacing_map)
        self.op(time=self.time_range.num - 1, dt=self.model.critical_dt)

    @property
    def wavefield(self):
        """get rid of the sponge that attenuates the waves"""
        wf_data=self.u.data[:, self.model.nbl:-self.model.nbl, self.model.nbl:-self.model.nbl]
        #wf_data_normalize = wf_data / np.amax(wf_data)
        #framerate = np.int(np.ceil(wf_data.shape[0] / self.n_frames))
        #return wf_data_normalize[0::framerate, :, :]
        return wf_data

    def plot_wavefield(self, time:int):
        """
        Gives a plot of the wavefield in time (ms)
        Args:
            time: value in ms
        Returns:

        """
        import matplotlib as cm
        fig = plt.figure(figsize=(15, 5))
        extent = [self.model.origin[0], self.model.origin[0] + 1e-3 * self.model.shape[0] * self.model.spacing[0],
                  self.model.origin[1] + 1e-3 * self.model.shape[1] * self.model.spacing[1], self.model.origin[1]]

        data_param = dict(vmin=-1e0, vmax=1e0, cmap=cm.Greys, aspect=1, extent=extent, interpolation='none')
        model_param = dict(vmin=1.5, vmax=2.5, cmap=cm.GnBu, aspect=1, extent=extent, alpha=.3)

        ax0 = fig.add_subplot(111)
        _ = plt.imshow(np.transpose(self.u.data[time, 40:-40, 40:-40]), **data_param)
        #_ = plt.imshow(np.transpose(vp), **model_param)
        ax0.set_ylabel('Depth (km)', fontsize=20)
        ax0.text(0.5, 0.08, "t = {:.0f} ms".format(time[times[0]]), ha="center", color='k')

        ax1 = fig.add_subplot(132)
        _ = plt.imshow(np.transpose(u.data[times[1], 40:-40, 40:-40]), **data_param)
        _ = plt.imshow(np.transpose(vp), **model_param)
        ax1.set_xlabel('X position (km)', fontsize=20)
        ax1.set_yticklabels([])
        ax1.text(0.5, 0.08, "t = {:.0f} ms".format(time[times[1]]), ha="center", color='k')

        ax2 = fig.add_subplot(133)
        _ = plt.imshow(np.transpose(u.data[times[2], 40:-40, 40:-40]), **data_param)
        _ = plt.imshow(np.transpose(vp), **model_param)
        ax2.set_yticklabels([])
        ax2.text(0.5, 0.08, "t = {:.0f} ms".format(time[times[2]]), ha="center", color='k')

        plt.show()

    def simulate_seismic_topo(self, topo, circles_list, not_circles, vmax=5, vmin=1, f0=0.02500, dx=10, dy=10, t0=0, tn=700,
                              pmlthickness=40, sigma_x=2, sigma_y=2, n_frames=50):
        if circles_list == []:
            circles_list = [[int(topo.shape[0] / 2), int(topo.shape[1] / 2)]]
        circles = np.array(circles_list)

        topo = topo.astype(np.float32)
        topoRescale = self.scale_linear(topo, vmax, vmin)
        veltopo = self.smooth_topo(topoRescale, sigma_x, sigma_y)
        if not_circles != []:
            veltopo[not_circles] = vmax * 1.8

        # Define the model
        model = Model(vp=veltopo,  # A velocity model.
                      origin=(0, 0),  # Top left corner.
                      shape=veltopo.shape,  # Number of grid points.
                      spacing=(dx, dy),  # Grid spacing in m.
                      nbpml=pmlthickness)  # boundary layer.

        dt = model.critical_dt  # Time step from model grid spacing
        nt = int(1 + (tn - t0) / dt)  # Discrete time axis length
        time = np.linspace(t0, tn, nt)  # Discrete modelling time

        u = TimeFunction(name="u", grid=model.grid,
                         time_order=2, space_order=2,
                         save=True, time_dim=nt)
        pde = model.m * u.dt2 - u.laplace + model.damp * u.dt
        stencil = Eq(u.forward, solve(pde, u.forward)[0])

        src_coords = np.multiply(circles[0], [dx, dy])
        src = RickerSource(name='src0', grid=model.grid, f0=f0, time=time, coordinates=src_coords)
        src_term = src.inject(field=u.forward, expr=src * dt ** 2 / model.m, offset=model.nbpml)

        if circles.shape[0] > 1:
            for idx, row in enumerate(circles[1:, :]):
                namesrc = 'src' + str(idx + 1)
                src_coords = np.multiply(row, [dx, dy])
                src_temp = RickerSource(name=namesrc, grid=model.grid, f0=f0, time=time, coordinates=src_coords)
                src_term_temp = src_temp.inject(field=u.forward, expr=src * dt ** 2 / model.m, offset=model.nbpml)
                src_term += src_term_temp

        op_fwd = Operator([stencil] + src_term)
        op_fwd(time=nt, dt=model.critical_dt)

        wf_data = u.data[:, pmlthickness:-pmlthickness, pmlthickness:-pmlthickness]
        wf_data_normalize = wf_data / np.amax(wf_data)

        framerate = np.int(np.ceil(wf_data.shape[0] / n_frames))
        return wf_data_normalize[0::framerate, :, :]

    def overlay_seismic_topography(self, image_in, wavefield_cube, time_slice, mask_flag=0, thrshld=.01, outfile=None):

        topo_image = plt.imread(image_in)

        if topo_image.shape[:2] != wavefield_cube.shape[1:]:
            wavefield = np.transpose(wavefield_cube[time_slice, :, :])
            if topo_image.shape[:2] != wavefield.shape:
                print("Topography shape does not match the wavefield shape")
        else:
            wavefield = wavefield_cube[time_slice, :, :]

        fig = plt.figure(figsize=(topo_image.shape[1] / 100, topo_image.shape[0] / 100), dpi=100, frameon=False)
        #    fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        data_param = dict(vmin=-.1e0, vmax=.1e0, cmap=cm.seismic, aspect=1, interpolation='none')

        if mask_flag == 0:
            waves = wavefield
            ax = plt.imshow(topo_image)
            ax = plt.imshow(waves, alpha=.4, **data_param)
        else:
            waves = np.ma.masked_where(np.abs(wavefield) <= thrshld, wavefield)
            ax = plt.imshow(topo_image)
            ax = plt.imshow(waves, **data_param)

        if outfile == None:
            plt.show()
            plt.close()
        else:
            plt.savefig(outfile, pad_inches=0)
            plt.close(fig)

    def show_widgets(self):
        pass


