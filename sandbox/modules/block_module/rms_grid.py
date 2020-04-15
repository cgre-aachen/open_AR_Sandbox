import numpy
import scipy
import pickle

class RMS_Grid():

    def __init__(self):
        """ Class to load RMS grids and convert them to a regular grid to use them in the Block module
        """

        self.nx = None
        self.ny = None
        self.nz = None
        self.block_dict = {}
        self.regular_grid_dict = {}
        # default resolution for the regriding. default is kinect v2 resolution and 100 depth levels
        self.regriding_resolution = [424, 512, 100]
        self.coords_x = None  # arrays to store coordinates of cells
        self.coords_y = None
        self.coords_z = None
        self.data_mask = None  # stores the Livecell information from the VIP  File
        self.reservoir_topography = None
        self.method = 'nearest'
        self.mask_method = 'nearest'

    def load_model_vip(self, infile):
        # parse the file
        f = open(infile, "r")

        while True:  # skip header
            l = f.readline().split()
            if len(l) > 2 and l[1] == "Size":
                break

        # n cells
        l = f.readline().split()
        self.nx = int(l[1])
        self.ny = int(l[2])
        self.nz = int(l[3])
        print('nx ny, nz:')
        print(self.nx, self.ny, self.nz)

        while True:  # skip to coordinates
            l = f.readline().split()
            if len(l) > 0 and l[0] == "CORP":
                print("loading cell positions")
                self.parse_coordinates(f, self.nx, self.ny, self.nz)
                print("coordinates loaded")
                break

        while True:  # skip to Livecell
            l = f.readline().split()
            if len(l) > 0 and l[0] == "LIVECELL":
                self.parse_livecells_vip(f, self.nx, self.ny, self.nz)
                print("Livecells loaded")
                break

        # parse the data
        while True:  # skip to key
            line = f.readline()
            l = line.split()
            if line == '':  # check if the end of file was reached and exit the loop if this is the case
                print('end of file reached')
                break

            elif len(l) >= 2 and l[1] == "VALUE":
                key = l[0]
                try:
                    # parse one block of data and store irt under the given key in the dictionary
                    self.parse_block_vip(f, self.block_dict, key, self.nx, self.ny, self.nz)
                except:
                    print('loading block "' + key + "' failed: not a valid VALUE Format")
                    break

        f.close()  # close the file

    def parse_coordinates(self, current_file, nx, ny, nz):
        f = current_file

        self.coords_x = numpy.empty((nx, ny, nz))
        self.coords_y = numpy.empty((nx, ny, nz))
        self.coords_z = numpy.empty((nx, ny, nz))

        for z in range(nz):

            print('processing coordinates in layer ' + str(z))
            for i in range(3):  # skip Layer(nz)
                f.readline()

            for y in range(ny):
                # print(y)
                for x in range(nx):

                    # skip cell header (each cell)
                    l = f.readline().split()
                    while l[0] == 'C':  # skip header
                        l = f.readline().split()

                    px = []
                    py = []
                    pz = []
                    for i in range(4):
                        # read the corner points
                        px.append(float(l[0]))
                        py.append(float(l[1]))
                        pz.append(float(l[2]))

                        px.append(float(l[3]))
                        py.append(float(l[4]))
                        pz.append(float(l[5]))
                        l = f.readline().split()  # read in next line

                    # calculate the arithmetic mean of all 4 corners elementwise:
                    self.coords_x[x, y, z] = numpy.mean(numpy.array(px))
                    self.coords_y[x, y, z] = numpy.mean(numpy.array(py))
                    self.coords_z[x, y, z] = numpy.mean(numpy.array(pz))

    def parse_livecells_vip(self, current_file, nx, ny, nz):
        data_np = numpy.empty((nx, ny, nz))

        # store pointer position to come back to after the values per line were determined
        pointer = current_file.tell()
        line = current_file.readline().split()
        values_per_line = len(line)
        # print(values_per_line)
        current_file.seek(pointer)  # go back to pointer position

        for z in range(nz):
            for y in range(ny):
                x = 0
                for n in range(nx // values_per_line):  # read values in full lines
                    l = current_file.readline().split()
                    if len(l) < values_per_line:  # if there is an empty line, skip to the next
                        l = current_file.readline().split()
                    for i in range(values_per_line):  # iterate values in the line
                        value = l[i]
                        data_np[x, y, z] = float(value)
                        x = x + 1  # iterate x

                if nx % values_per_line > 0:
                    l = current_file.readline().split()
                    for i in range(nx % values_per_line):  # read values in the last not full line
                        value = l[i]
                        data_np[x, y, z] = float(value)
                        x = x + 1

        self.block_dict['mask'] = data_np

    def parse_block_vip(self, current_file, value_dict, key, nx, ny, nz):
        data_np = numpy.empty((nx, ny, nz))

        f = current_file

        pointer = f.tell()  # store pointer position to come back to after the values per line were determined
        for i in range(3):  # skip header
            f.readline()

        l = f.readline().split()
        values_per_line = len(l)
        # print('values per line: ' + str(values_per_line))
        blocklength = nx // values_per_line
        f.seek(pointer)  # go back to pointer position

        # read block data
        if (nx % values_per_line) != 0:
            blocklength = blocklength + 1

        for z in range(nz):
            for i in range(3):
                l = f.readline().split()
            for y in range(ny):
                x = 0

                for line in range(blocklength):
                    l = f.readline().split()
                    if len(l) < 1:
                        l = f.readline().split()  # skip empty line that occurs if value is dividable by 8
                    while l[0] == "C":
                        l = f.readline().split()  # skip the header lines(can vary from file to file)
                    for i in range(len(l)):
                        try:
                            value = l[i]
                            # data.loc[x,y,z] = value
                            # values.append(value)
                            data_np[x, y, z] = float(value)
                            x = x + 1
                        except:
                            print('failed to parse value ', x, y, z)
                            print(l)
                            x = x + 1
        # print(x, y + 1, z + 1)  # to check if all cells are loaded

        print(key + ' loaded')
        value_dict[key] = data_np

        return True

    def convert_to_regular_grid(self, method=None, mask_method=None):
        # prepare the cell coordinates of the original grid
        x = self.coords_x.ravel()
        y = self.coords_y.ravel()
        z = self.coords_z.ravel()

        # prepare the coordinates of the regular grid cells:
        # define extent:
        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()
        zmin = z.min()
        zmax = z.max()

        # prepare the regular grid:
        gx = numpy.linspace(xmin, xmax, num=self.regriding_resolution[0])
        gy = numpy.linspace(ymin, ymax, num=self.regriding_resolution[1])
        gz = numpy.linspace(zmin, zmax, num=self.regriding_resolution[2])

        a, b, c = numpy.meshgrid(gx, gy, gz)

        grid = numpy.stack((a.ravel(), b.ravel(), c.ravel()), axis=1)

        # iterate over all loaded datasets:
        for key in self.block_dict.keys():
            print("processing grid: ", key)
            if key == 'mask':
                self.block_dict[key][:, :, 0] = 0.0
                self.block_dict[key][0, :,
                :] = 0.0  # exchange outer limits of the box so that nearest neighbour returns zeros outside the box
                self.block_dict[key][-1, :, :] = 0.0
                self.block_dict[key][:, -1, :] = 0.0
                self.block_dict[key][:, 0, :] = 0.0
                self.block_dict[key][:, :, -1] = 0.0
                self.block_dict[key][:, :, 0] = 0.0

            data = self.block_dict[key].ravel()

            if key == 'mask':  # for the mask, fill NaN values with 0.0
                if mask_method == None:
                    mask_method = self.mask_method  # 'linear' or 'nearest'
                data = numpy.nan_to_num(data)  # this does not work with nearest neighbour!

                interp_grid = scipy.interpolate.griddata((x, y, z), data, grid, method=mask_method)

            else:
                if method == None:
                    method = self.method
                interp_grid = scipy.interpolate.griddata((x, y, z), data, grid, method=method)

            # save to dictionary:
            # reshape to originasl dimension BUT WITH X AND Y EXCHANGEND
            self.regular_grid_dict[key] = interp_grid.reshape([self.regriding_resolution[1],
                                                               self.regriding_resolution[0],
                                                               self.regriding_resolution[2]]
                                                              )
            print("done!")

    def create_reservoir_topo(self):
        """
        creates a 2d array with the z values of the reservoir top (the z coordinate of the top layer in the array
        """
        # create 2d grid for lookup:
        x = self.coords_x.ravel()
        y = self.coords_y.ravel()

        # prepare the coordinates of the regular grid cells:
        # define extent:
        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()

        # prepare the regular grid:
        gx = numpy.linspace(xmin, xmax, num=self.regriding_resolution[0])
        gy = numpy.linspace(ymin, ymax, num=self.regriding_resolution[1])
        a, b = numpy.meshgrid(gx, gy)

        grid2d = numpy.stack((a.ravel(), b.ravel()), axis=1)

        top_x = self.coords_x[:, :, 0].ravel()
        top_y = self.coords_y[:, :, 0].ravel()
        top_z = self.coords_z[:, :, 0].ravel()

        topo = scipy.interpolate.griddata((top_x, top_y), top_z, grid2d)  # this has to be done with the linear method!
        self.reservoir_topography = topo.reshape([self.regriding_resolution[1], self.regriding_resolution[0]])

    def save(self, filename):
        """
    saves a list with two entries to a pickle:

        [0] the regridded data blocks in a dictionary
        [1] the reservoir topography map

        """
        pickle.dump([self.regular_grid_dict, self.reservoir_topography], open(filename, "wb"))

