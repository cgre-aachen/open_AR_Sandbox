from sandbox.modules.template import ModuleTemplate
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy import ndimage
from matplotlib import cm

from .model import Model
from .source import RickerSource, TimeAxis, Receiver
from devito import TimeFunction, Eq, solve, Operator


class SeismicModule(ModuleTemplate):
    """ Follow the link to see the code that inspired the module
    https://nbviewer.jupyter.org/github/devitocodes/devito/blob/master/examples/seismic/tutorials/01_modelling.ipynb"""
    def __init__(self, **kwargs):
        """Importing packages and setting correctly the path to the devito package, this beacuse we need the examples"""
        self.vp = None #Normalized or smoothed or None topography to be the velocity model
        self.model: Model = None #the model
        self.time_range = None #the time of the simulation
        self.src = None #the ricker source
        self.u = None #wavefield time function
        self.pde = None #PDE to solve. Wave equation with corresponding discretizations
        self.stencil= None # a time marching updating equation known as a stencil using customized SymPy functions
        self.src_coordinates=[]
        self.src_term = [] #all sources together
        self.rec=None
        self.rec_term = [] #all receivers
        self.op = None #aAfter constructing all the necessary expressions for updating the wavefield, injecting the source term and interpolating onto the receiver points, we can now create the Devito operator
        self.n_frames_speed = 10
        #self.aruco_sources_coord = []
        # For the cropping, because when normalizing the velocities dont work
        self.box_origin = [40, 40]  # location of bottom left corner of the box in the sandbox. values refer to pixels of the kinect sensor
        self.box_width = 200
        self.box_height = 150

        self.xy_aruco=[]
        # for the plotting
        self.timeslice = 0
        self.field = None
        self.p_velocity = True
        self.p_wave = True
        self.ready_velocity = False
        self.ready_wave = False
        self.framerate = 10
        self._vp = None
        self._w = None
        self.real_time = True

    @property
    def to_box_extent(self):
        """When using imshow to plot data over the image. pass this as extent argumment to display the
        image in the correct area of the sandbox box-area"""
        return (self.box_origin[0], self.box_width + self.box_origin[0],
                self.box_origin[1], self.box_height + self.box_origin[1])

    def update(self, sb_params: dict):
        frame = sb_params.get('frame')
        ax = sb_params.get('ax')
        marker = sb_params.get('marker')
        self.frame = frame[self.box_origin[1]:self.box_origin[1] + self.box_height,
                        self.box_origin[0]:self.box_origin[0] + self.box_width]
        if len(marker) > 0:
            self.xy_aruco = marker.loc[marker.is_inside_box, ('box_x', 'box_y')].values
        else:
            self.xy_aruco=[]

        ax = self.plot(ax)
        return sb_params

    def plot(self, ax):
        if self.p_velocity and self.ready_velocity:
            self.plot_seismic_velocity(ax)
        else:
            self.delete_seismic_velocity()
        if self.p_wave and self.ready_wave:
            self.plot_wavefield(ax)
        else:
            self.delete_wavefield()
        return ax

    def plot_seismic_velocity(self, ax):
        if self.field is not None:
            if self._vp is None:

                self._vp = ax.imshow(self.field, cmap=plt.get_cmap("jet"),
                          vmin=np.min(self.field), vmax=np.max(self.field),
                          extent=self.to_box_extent, origin="lower", zorder=2)
            else:
                self._vp.set_data(self.field)

    def delete_seismic_velocity(self):
        if self._vp is not None:
            self._vp.remove()
            self._vp = None

    def plot_wavefield(self, ax):
        if self.u is not None and self.model is not None:
            if self._w is None:
                self._w = ax.imshow(self.wavefield(self.timeslice),
                          vmin=-1e0, vmax=1e0,
                          cmap=plt.get_cmap('seismic'),
                          aspect=1, extent=self.to_box_extent,
                          interpolation='none', origin="lower", zorder=3)
            else:
                self._w.set_data(self.wavefield(self.timeslice))

            if self.real_time:
                self.timeslice += self.framerate
                if self.timeslice >= self.time_range.num:
                    self.timeslice = 0


    def delete_wavefield(self):
        if self._w is not None:
            self._w.remove()
            self._w = None

    def run_simulation(self, vmin=2, vmax=4, nbl=40, **kwargs):
        if self.model is None:
            self.init_model(vmin=vmin, vmax=vmax, frame=self.frame, nbl=nbl, **kwargs)
        if len(self.src_coordinates)==0:
            return print("Put an aruco marker as a source or manually specify a source term")
        self.insert_aruco_source()
        self.operator_and_solve()
        return print("Simulation succesfull")

    def init_model(self, vmin, vmax, frame=None, **kwargs):
        if frame is None:
            frame = self.frame
        self.create_velocity_model(frame,
                                   vmax=vmax,
                                   vmin=vmin,
                                   **kwargs)
        self.create_time_axis(**kwargs)
        self.create_time_function()
        self.solve_PDE()

    def insert_aruco_source(self):
        if self.model is None:
            return print("Create the velocity model first")

        if len(self.xy_aruco)>0:
            self.src_term=[]
            for counter, aru in enumerate(self.xy_aruco):
                #Modify to the simulation area
                aru_mod = (aru[0]-self.box_origin[0], aru[1]-self.box_origin[1])
                src = self.create_source(name="src%i"%counter, f0=0.025,
                                         source_coordinates=(aru_mod[0]*self.model.spacing[0],aru_mod[1]*self.model.spacing[1]),
                                                             show_wavelet=False,show_model=False)
                self.inject_source(src)
        else:
            return print("No arucos founded")

    def crop_frame(self, origin:tuple, height:int, width:int, frame=None):
        self.box_origin = origin # location of bottom left corner of the box in the sandbox. values refer to pixels of the kinect sensor
        self.box_width = width
        self.box_height = height
        if frame is not None:
            frame = frame[self.box_origin[1]:self.box_origin[1] + self.box_height,
                        self.box_origin[0]:self.box_origin[0] + self.box_width]
            return frame

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
                              vmin:float=2.0, smooth:bool=False, sigma_x:int=2, sigma_y:int=2,
                              spacing:tuple = (10, 10), origin=(0,0),
                              nbl:int=40, show_velocity: bool=False, **kwargs):
        """
        takes the topography and creates a velocity model from the topography
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
            #Make a simple 2 layered velocity model
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
        self.model = Model(vp=topo, origin=origin, shape=topo.shape, spacing=spacing,
              space_order=2, nbl=nbl, bcs="damp")
        if show_velocity:
            self.show_velocity(self.model)
        slices = tuple(slice(self.model.nbl, -self.model.nbl) for _ in range(2))
        if getattr(self.model, 'vp', None) is not None:
            self.field = self.model.vp.data[slices]
        else:
            self.field = self.model.lam.data[slices]
        self.ready_velocity = True
        return self.model

    def create_time_axis(self, t0:int=0, tn:int=1000, **kwargs):
        """
        Time duration of our model. This takes the start, and end with the timestepsize provided by the model
        Args:
            t0:  Simulation starts a t=0
            tn:  Simulation last 1 second (1000 ms)
        Returns:
            Time_range
        """
        dt = self.model.critical_dt
        self.time_range = TimeAxis(start=t0, stop=tn, step=dt)
        return self.time_range

    @property
    def _src_coords(self):
        """gives the source coordinates of the richter waveletet in the middle of the model"""
        #src_coords = np.multiply(circles[0], [dx, dy])
        return np.multiply([int(self.model.domain_size[0]/2),int(self.model.domain_size[1]/2/2)],
                                 [10, 10])

    def create_source(self, name:str = 'src', f0: float=0.025, source_coordinates:tuple=None,
                      show_wavelet:bool=False, show_model:bool=False):
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
            source_coordinates = self._src_coords

        src = RickerSource(name=name, grid=self.model.grid, f0=f0,
                                npoint=1, time_range=self.time_range, coordinates=source_coordinates)
        ## First, position source centrally in all dimensions, then set depth
        #src.coordinates.data[0, :] = np.array(model.domain_size) * .5
        #src.coordinates.data[0, -1] = 20.  # Depth is 20m
        if show_wavelet: # We can plot the time signature to see the wavelet
            src.show()
        if show_model:
            self.show_velocity(self.model, source=src.coordinates.data)
        return src

    def create_time_function(self):
        """
        second order time discretization: This derivative is represented in Devito by u.dt2 where u is a TimeFunction object.
        Spatial discretization: This derivative is represented in Devito by u.laplace where u is a TimeFunction object.
        With space and time discretization defined, we can fully discretize the wave-equation with the combination of time and space discretizations
        Returns:

        """
        # Define the wavefield with the size of the model and the time dimension
        self.u = TimeFunction(name="u", grid=self.model.grid,
                              time_order=2, space_order=2,
                                   save=self.time_range.num)
        # We can now write the PDE
        self.pde = self.model.m * self.u.dt2 - self.u.laplace + self.model.damp * self.u.dt
        return self.pde

    def solve_PDE(self):
        """This discrete PDE can be solved in a time-marching way updating u(t+dt) from the previous time step"""
        self.stencil = Eq(self.u.forward, solve(self.pde, self.u.forward))
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
        self.src_coordinates.append(source.coordinates.data)
        return self.src_term

    def create_receivers(self, name:str = 'rec', n_receivers:int=50, depth_receivers:int=20, show_receivers:bool=False):
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
        rec = Receiver(name=name, npoint=n_receivers, time_range=self.time_range,
                        grid=self.model.grid, coordinates=rec_coords)
        if show_receivers:
            self.show_velocity(self.model, receiver=rec.coordinates.data[::4, :])
        return rec

    def interpolate_receiver(self, rec):
        """obtain the receivers information"""
        self.rec_term = rec.interpolate(expr=self.u.forward)
        return self.rec_term

    def operator_and_solve(self):
        """
        After constructing all the necessary expressions for updating the wavefield, injecting the source term and
        interpolating onto the receiver points, we can now create the Devito operator that will generate the C code at runtime.
        Returns:

        """
        self.op = Operator([self.stencil] + self.src_term + self.rec_term)#, subs=self.model.spacing_map)
        self.op(dt=self.model.critical_dt)
        self.ready_wave = True

    def wavefield(self, timeslice, thrshld=0.01):
        """get rid of the sponge that attenuates the waves"""
        #wf_data=self.u.data[:, self.model.nbl:-self.model.nbl, self.model.nbl:-self.model.nbl]
        if timeslice>=self.time_range.num:
            print('timeslice not valid for value %i, setting values of %i' %(timeslice,self.time_range.num-1))
            timeslice=self.time_range.num-1

        wf_data = self.u.data[timeslice, self.model.nbl:-self.model.nbl, self.model.nbl:-self.model.nbl]
        wf_data_normalize = wf_data / np.amax(wf_data)
        waves = np.ma.masked_where(np.abs(wf_data_normalize) <= thrshld, wf_data_normalize)
        return waves

    def show_wavefield(self, timeslice:int):
        """
        Gives a plot of the wavefield in time (ms)
        Args:
            timeslice: value in ms
        Returns:

        """

        fig, ax = plt.subplots()
        domain_size = 1.e-3 * np.array(self.model.domain_size)
        extent = [self.model.origin[0], self.model.origin[0] + domain_size[0],
                  self.model.origin[1] + domain_size[1], self.model.origin[1]]

        model_param = dict(vmin=self.vp.max(), vmax=self.vp.min(), cmap=plt.get_cmap('jet'), aspect=1, extent=extent, origin="lower")
        data_param = dict(vmin=-1e0, vmax=1e0, cmap=plt.get_cmap('seismic'), aspect=1, extent=extent, interpolation='none')#, alpha=.4)

        _vp = plt.imshow(np.transpose(self.vp), **model_param)
        _wave = plt.imshow(np.transpose(self.wavefield(timeslice)), **data_param)
        ax.set_ylabel('Depth (km)', fontsize=20)
        ax.set_xlabel('x position (km)', fontsize=20)

        ax.set_title("t = {:.0f} ms".format((timeslice)*self.time_range.step))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(_vp, cax=cax, ax=ax, label="Velocity (km/s)")
        #fig.colorbar(_vp, ax=ax, label="Wave amplitude")
        fig.show()

    def show_velocity(self, model, source=None, receiver=None, cmap="jet"):
        """
        Function modified from DEVITO example codes
        Plot a two-dimensional velocity field from a seismic `Model`
        object. Optionally also includes point markers for sources and receivers.

        Parameters
        ----------
        model : Model
            Object that holds the velocity model.
        source : array_like or float
            Coordinates of the source point.
        receiver : array_like or float
            Coordinates of the receiver points.
        colorbar : bool
            Option to plot the colorbar.
        """
        domain_size = 1.e-3 * np.array(model.domain_size)
        extent = [model.origin[0], model.origin[0] + domain_size[0],
                  model.origin[1] + domain_size[1], model.origin[1]]

        slices = tuple(slice(model.nbl, -model.nbl) for _ in range(2))
        if getattr(model, 'vp', None) is not None:
            field = model.vp.data[slices]
        else:
            field = model.lam.data[slices]
        fig, ax = plt.subplots()
        plot = ax.imshow(field, animated=True, cmap=cmap,
                          vmin=np.min(field), vmax=np.max(field),
                          extent=extent, origin="lower")
        ax.set_xlabel('X position (km)')
        ax.set_ylabel('Depth (km)')

        # Plot source points, if provided
        if receiver is not None:
            ax.scatter(1e-3 * receiver[:, 0], 1e-3 * receiver[:, 1],
                        s=25, c='green', marker='D')

        # Plot receiver points, if provided
        if source is not None:
            if not isinstance(source, list):
                source = [source]
            for sou in source:
                ax.scatter(1e-3 * sou[:, 0], 1e-3 * sou[:, 1],
                            s=25, c='red', marker='o')

        # Ensure axis limits
        ax.set_xlim(model.origin[0], model.origin[0] + domain_size[0])
        ax.set_ylim(model.origin[1] + domain_size[1], model.origin[1])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(plot, cax=cax, ax=ax, label="Velocity (km/s)")
        fig.show()

    def show_shotrecord(self, rec, model, t0, tn):
        """
        function modified from DEVITO example codes
        Plot a shot record (receiver values over time).

        Parameters
        ----------
        rec :
            Receiver data with shape (time, points).
        model : Model
            object that holds the velocity model.
        t0 : int
            Start of time dimension to plot.
        tn : int
            End of time dimension to plot.
        """
        scale = np.max(rec) / 10.
        extent = [model.origin[0], model.origin[0] + 1e-3 * model.domain_size[0],
                  1e-3 * tn, t0]
        fig, ax = plt.subplots()
        plot = ax.imshow(rec, vmin=-scale, vmax=scale, cmap=cm.gray, extent=extent)
        ax.set_xlabel('X position (km)')
        ax.set_ylabel('Time (s)')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(plot, cax=cax, ax =ax)
        fig.show()


    def show_widgets(self):
        pass


