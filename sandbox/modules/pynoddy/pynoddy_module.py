import numpy as np
import pyvista
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sandbox.modules.template import ModuleTemplate
from sandbox.modules.gempy.utils import Grid
from pynoddy.output import NoddyOutput
from sandbox import set_logger
logger = set_logger(__name__)


class PynoddyModule(ModuleTemplate):
    """https://github.com/cgre-aachen/pynoddy"""

    def __init__(self, output_model: NoddyOutput = None, extent: list = None, box: list = None):
        """

        Args:
            output_model:
            extent:
            box:
        """

        self.output_model = output_model
        self.sensor_extent = extent
        self.box_dimensions = box

        self.block_model = None
        self.model_extent = None
        self.model_resolution = None
        self.model_spacing = None
        self.grid = None

        self._block = None
        self._values_ix = None
        self.mask = None

        self.vertices_mapview = None

        self.set_NoddyOutput(self.output_model)

        self.hill = None
        self.lith = None
        self.lock = None

        logger.info("PynoddyModule loaded successfully")

    def set_NoddyOutput(self, n: NoddyOutput):
        """
        Args:
            n:

        Returns:
        """
        self.output_model = n
        self.model_extent = list(map(int, [0, n.extent_x, 0, n.extent_y, 0, n.extent_z]))
        logger.info("Model extent: %s" % self.model_extent)
        self.model_resolution = list(map(int, [n.nx, n.ny, n.nz]))
        logger.info("Model resolution: %s" % self.model_resolution)
        self.model_spacing = list(map(int, [n.delx, n.dely, n.delz]))
        logger.info("Size of each block: %s" % self.model_spacing)
        self.block_model = n.block
        self.grid = Grid(self.box_dimensions, self.model_extent, [0, self.model_resolution[0],
                                                                  0, self.model_resolution[1],
                                                                  self.sensor_extent[-2], self.sensor_extent[-1]])
        self.create_empty_block()

    def update(self, sb_params: dict):
        frame = sb_params.get("frame")
        extent = sb_params.get("extent")
        ax = sb_params.get("ax")
        self.lock = sb_params.get('lock_thread')

        scale_frame = self.scale_frame_to_model(frame)
        _ = self.grid.update_grid(scale_frame)

        empty2d = np.zeros((self.model_resolution[0], self.model_resolution[1], 3))
        for i in range(3):
            empty2d[:, :, i] = self.grid.depth_grid[:, i].reshape(self.model_resolution[:2])
        topo_level = empty2d[..., 2, np.newaxis]

        self.create_topography_mask(topo_level)
        self.set_block_solution_to_topography()
        self.plot(scale_frame, ax)

        sb_params['ax'] = ax
        sb_params['frame'] = scale_frame
        #sb_params['cmap'] = cmap
        #sb_params['marker'] = self.modelspace_arucos
        # This because we are currently plotting our own cmap and shading
        sb_params['active_cmap'] = False
        sb_params['active_shading'] = False
        sb_params['extent'] = self.model_extent
        # sb_params['del_contour'] = not self.show_boundary

        return sb_params

    def create_empty_block(self):
        dx = np.arange(self.model_extent[0], self.model_extent[1], self.model_spacing[0])
        dy = np.arange(self.model_extent[2], self.model_extent[3], self.model_spacing[1])
        dz = np.arange(self.model_extent[4], self.model_extent[5], self.model_spacing[2])

        g = np.meshgrid(dx, dy, dz, indexing="ij")
        values = np.vstack(tuple(map(np.ravel, g))).T.astype("float64")
        self._values_ix = values
        self._block = values[:, 2].reshape(self.model_resolution)

    def create_topography_mask(self, topo_level):
        self.mask = np.greater(topo_level, self._block)

    def set_block_solution_to_topography(self):
        """

        Returns:

        """
        cop = self.block_model.copy()
        mask = np.where(cop * self.mask == 0)
        cop[mask] = -1

        height, width, depth = cop.shape
        vertices = []
        for i in range(height):
            for j in range(width):
                pos = np.argmin(cop[i, j, :])
                vertices.append([i, j, self.block_model[i, j, pos]])
        self.vertices_mapview = np.asarray(vertices)

    def plot_3D(self, topography=True, notebook=True, **kwargs):
        if topography and self.mask is not None:
            new_block = self.block_model.copy() * self.mask
            new_block[np.where(new_block == 0)] = np.nan
        else:
            new_block = self.block_model.copy()
        cmap_type = kwargs.pop('cmap', 'YlOrRd')
        pyvista.plot(new_block, notebook=notebook, cmap = cmap_type, **kwargs)

    def plot_section(self, direction='y', position='center', topography=True, colorbar=True, **kwargs):
        """
         Create asecion block through the model
        Args:
            direction: 'x', 'y', 'z' : coordinate direction of section plot (default: 'y')
            position:  int or 'center' : cell position of section as integer value
            topography:
            colorbar:
            **kwargs:

        Returns:

        """
        aspect = kwargs.pop("aspect", "auto")
        cmap_type = kwargs.pop('cmap', 'YlOrRd')

        if 'ax' in kwargs:
            # append plot to existing axis
            ax = kwargs.pop('ax')
        else:
            figsize = kwargs.pop("figsize", (10, 6))
            fig, ax = plt.subplots(figsize=figsize)

        if 'x' in direction:
            xlabel = "y"
            ylabel = "z"
        elif 'y' in direction:
            xlabel = "x"
            ylabel = "z"
        elif 'z' in direction:
            xlabel = "x"
            ylabel = "y"

        if topography and self.mask is not None:
            data = self.block_model.copy() * self.mask
            data[np.where(data==0)] = np.nan
        else:
            data = self.block_model.copy()

        # plot section
        section_slice, cell_pos = self.output_model.get_section_voxels(direction, position, data=data)
        title = kwargs.pop("title", "Section in %s-direction, pos=%d" % (direction, cell_pos))

        im = ax.imshow(section_slice, interpolation='nearest', aspect=aspect, cmap=cmap_type, origin='lower', **kwargs)

        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            ax.figure.colorbar(im, cax=cax, ax=ax, label='Lithology')

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax

    def delete_ax(self, ax):
        """
        replace the ax.cla(). delete contour fill and images of hillshade and lithology
        Args:
            ax:
        Returns:
            ax
        """
        if self.lith is not None:
            self.lith.remove()
            self.lith = None
        if self.hill is not None:
            self.hill.remove()
            self.hill = None

        [fill.remove() for fill in reversed(ax.collections) if isinstance(fill, matplotlib.collections.PathCollection)]
        [text.remove() for text in reversed(ax.texts) if isinstance(text, matplotlib.text.Text)]
        [coll.remove() for coll in reversed(ax.collections) if isinstance(coll, matplotlib.collections.LineCollection)]
        return ax

    def plot_mapview(self,
                     show_lith: bool = True,
                     # show_boundary: bool = True,
                     show_hillshade: bool = True,
                     show_contour: bool = False,
                     show_only_faults: bool = False,
                     **kwargs):

        cmap_type = kwargs.pop('cmap', 'YlOrRd')
        if 'ax' in kwargs:
            # append plot to existing axis
            ax = kwargs.pop('ax')
        else:
            figsize = kwargs.pop("figsize", (10, 6))
            fig, ax = plt.subplots(figsize=figsize)

        if show_lith:
            image = self.vertices_mapview[:, 2].reshape(self.model_resolution[:2])
            self.lith = ax.imshow(image,
                             origin='lower',
                             zorder=-10,
                             extent=self.model_extent[:4],
                             cmap=cmap_type,
                             #norm=norm,
                             aspect='auto')



        fill_contour = kwargs.pop('show_fill_contour', False)
        azdeg = kwargs.pop('azdeg', 0)
        altdeg = kwargs.pop('altdeg', 0)
        super = kwargs.pop('super_res', False)
        colorbar = kwargs.pop("show_colorbar", False)

        topo = self.grid.depth_grid[:, 2].reshape(self.model_resolution[:2])
        #if super:
        #    import skimage
        #    topo_super_res = skimage.transform.resize(
        #        topo,
        #        (1600, 1600),
        #        order=3,
        #        mode='edge',
        #        anti_aliasing=True, preserve_range=False)
        #    values = topo_super_res[..., 2]
        #else:
        #    values = topo.values_2d[..., 2]

        if show_contour is True:
            CS = ax.contour(topo, extent=self.model_extent[:4],
                            colors='k', linestyles='solid', origin='lower')
            ax.clabel(CS, inline=1, fontsize=10, fmt='%d')
        if fill_contour is True:
            CS2 = ax.contourf(topo, extent=self.model_extent[:4], cmap=cmap)
            if colorbar:
                from gempy.plot.helpers import add_colorbar
                add_colorbar(axes=ax, label='elevation [m]', cs=CS2)

        if show_hillshade:
            from matplotlib.colors import LightSource
            # Note: 180 degrees are subtracted because visualization in Sandbox is upside-down
            ls = LightSource(azdeg=azdeg - 180, altdeg=altdeg)
            # TODO: Is it better to use ls.hillshade or ls.shade??
            hillshade_topography = ls.hillshade(topo)
            # vert_exag=0.3,
            # blend_mode='overlay')
            self.hill = ax.imshow(hillshade_topography,
                             cmap=plt.cm.gray,
                             origin='lower',
                             extent=self.model_extent[:4],
                             alpha=0.4,
                             zorder=11,
                             aspect='auto')

    def plot(self, frame, ax):
        self.delete_ax(ax)
        self.plot_mapview(ax=ax)

    def scale_frame_to_model(self, topo):
        """
        Get the sandbox frame and rescale it to the model extents
        Args:
            topo:
        Returns:

        """
        grid_topo = skimage.transform.resize(topo,
                                             (self.model_resolution[:2]),
                                             mode='constant',
                                             anti_aliasing=False,
                                             preserve_range=True)
        scale_frame = self.grid.scale_frame(grid_topo)
        return scale_frame

    def show_widgets(self):
        pass
