import numpy as np
import pyvista
import skimage.transform
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

        scale_frame = self.scale_frame_to_model(frame)
        _ = self.grid.update_grid(scale_frame)

        empty2d = np.zeros((self.model_resolution[0], self.model_resolution[1], 3))
        for i in range(3):
            empty2d[:, :, i] = self.grid.depth_grid[:, i].reshape(self.model_resolution[:2])
        topo_level = empty2d[..., 2, np.newaxis]
        self.create_topography_mask(topo_level)
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

    def plot_3D(self, topography=True, notebook=True, **kwargs):
        if topography and self.mask is not None:
            new_block = self.block_model.copy() * self.mask
            new_block[np.where(new_block == 0)] = np.nan
        else:
            new_block = self.block_model.copy()
        pyvista.plot(new_block, notebook=notebook, kwargs=kwargs)

    def plot_section(self, direction='y', position='center', topography=True, colorbar=True, **kwargs):
        """
         Create asecion block through the model
        Args:
            direction: 'x', 'y', 'z' : coordinate direction of section plot (default: 'y')
            position:  int or 'center' : cell position of section as integer value
            **kwargs:

        Returns:

        """
        ve = kwargs.pop("ve", "auto")
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

        im = ax.imshow(section_slice, interpolation='nearest', aspect=ve, cmap=cmap_type, origin='lower', **kwargs)

        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            ax.figure.colorbar(im, cax=cax, ax=ax, label='Lithology')

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return ax

    def plot_mapview(self, lithology=True, contours=True, **kwargs):
        img = self.vertices_mapview[:,2].reshape(self.model_resolution[:2]).T
        if 'ax' in kwargs:
            # append plot to existing axis
            ax = kwargs.pop('ax')
        else:
            figsize = kwargs.pop("figsize", (10, 6))
            fig, ax = plt.subplots(figsize=figsize)





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


    def plot(self, frame, ax):
        pass

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