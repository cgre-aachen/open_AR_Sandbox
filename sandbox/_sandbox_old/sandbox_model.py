import annoy
import numpy as np
from tqdm.auto import tqdm
import xarray

# TODO: Automatically add mask to grid model?
# TODO: Where to map data values to grid?


class Resampling(object):
    """ Methods to resample models on a regular grid
    """

    def __init__(self, grid):
        """

        Args:
            grid (Model): Model object, from which the coordinates are fetched
        """

        self.grid = grid
        self.data_lookup = annoy.AnnoyIndex(3, 'euclidean')  # dimensions and distance metric

        self.mask = None
        self.indices = None

    def build_data_lookup(self, data, tree_number):
        # add all data points to an annoy lookup object

        range_a = tqdm(np.arange(data.shape[0]), 'Building data lookup...')  # track process
        for i in range_a:
            self.data_lookup.add_item(i, data[i])

        tqdm.write('Building trees... (That can take a while)')
        self.data_lookup.build(tree_number)  # build lookup trees, afterwards items cannot be added anymore

    def save_lookup(self, filename='lookup.ann'):
        self.data_lookup.save(filename)

    def load_lookup(self, filename='lookup.ann'):
        lookup = annoy.AnnoyIndex(3, 'euclidean')
        lookup.load(filename)
        self.data_lookup = lookup


    def find_nearest_neighbours(self, threshold=10):
        """Find nearest neighbour of a new grid-cell in a set of data-grid-cells

        Args:
            data (array): n x 3 array with x,y,z coordinates of irregular grid points
            grid (array): n x 3 array with x,y,z coordinates of regular grid points
            threshold (float): Maximum distance, within a neighbour is accepted as such

        Returns:
            mask (bool array): 1D mask defining the validity of grid cells dependent on threshold
            idx (int array): 1D array of size grid[mask], with inidces pointing to nearest point in data
            mindist (float array): 1D array of size grid[mask], with distances to indixed neighbour (for testing)
        """

        grid = self.grid.get_coords()

        idx = np.empty(grid.shape[0], dtype=np.int)
        mindist = np.empty(grid.shape[0])

        # start lookup for each grid point
        range_b = tqdm(np.arange(grid.shape[0]), 'Calculating distances...')
        for i in range_b:
            # lookup of nearest neighbours for each grid point, get index and distance
            result = self.data_lookup.get_nns_by_vector(grid[i], 1, include_distances=True)
            idx[i] = result[0][0]  # minimum distance index for each grid point
            mindist[i] = result[1][0]  # minimum distance for each grid point

        # get "valid" distances, indices and the mask to filter grid
        # TODO: Move to seperate function
        if threshold is not None:
            tqdm.write('Creating and applying mask...')
            self.mask = np.where(mindist < threshold, True, False)
            #self.indices = idx[self.mask] don't do this here
            #mindist = mindist[mask]
        else:
            self.mask = np.ones(grid.shape[0], dtype=bool)

        self.indices = idx

        return True


class Model(object):

    def __init__(self, x_coords=None, y_coords=None, z_coords=None, filename=None):

        self.dataset = None

        if filename is not None:
            self.load(filename)
        else:
            self.dataset = xarray.Dataset(coords={'X': x_coords, 'Y': y_coords, 'Z': z_coords})

    @property
    def dimensions(self):
        return tuple(dict(self.dataset.dims.mapping).values())

    def add_attribute(self, name, array):
        if array.shape == self.dimensions:
            da = xarray.DataArray(array, dims=['X','Y','Z'])
            self.dataset = self.dataset.assign({name: da})
        else:
            print('Cannot assign array. Model dimensions are:', self.dimensions)

    def get_attribute(self, name):
        return self.dataset[name].values

    def list_attributes(self):
        return list(self.dataset.variables.mapping.keys())

    def get_coords(self):
        x, y, z = np.meshgrid(self.dataset.coords['X'].values,
                              self.dataset.coords['Y'].values,
                              self.dataset.coords['Z'].values)
        return np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)

    def save(self, filename='model.nc'):
        self.dataset.to_netcdf(filename)

    def load(self, filename):
        self.dataset = xarray.open_dataset(filename)


