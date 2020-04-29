import numpy

class Raw_landslides_simulation:
    def __init__(self):
        self.folder_dir_out = None

        self.ncols = None
        self.nrows = None
        self.xllcorner = None
        self.yllcorner = None
        self.cellsize = None
        self.NODATA_value = None
        self.asc_data = None

        self.a_line = None
        self.b_line = None
        self.xyz_data = None

        self.release_area = None
        self.hazard_map = None
        self.max_height = None
        self.max_velocity = None

        self.domain = None
        self.absolute_topo = None
        self.relative_topo = None

        self.horizontal_flow = None
        self.vertical_flow = None

        self.flow_selector = None
        self.frame_selector = 0
        self.counter = 1
        self.simulation_frame = 0
        self.running_simulation = False

        self.widget = None

        self.npz_filename = None

    def _load_data_asc(self, infile):
        f = open(infile, "r")
        self.ncols = int(f.readline().split()[1])
        self.nrows = int(f.readline().split()[1])
        self.xllcorner = float(f.readline().split()[1])
        self.yllcorner = float(f.readline().split()[1])
        self.cellsize = float(f.readline().split()[1])
        self.NODATA_value = float(f.readline().split()[1])
        self.asc_data = numpy.reshape(numpy.array([float(i) for i in f.read().split()]), (self.nrows, self.ncols))
        return self.asc_data


    def _load_data_xyz(self, infile):
        f = open(infile, "r")
        self.ncols, self.nrows = map(int, f.readline().split())
        self.a_line = numpy.array([float(i) for i in f.readline().split()])
        self.b_line = numpy.array([float(i) for i in f.readline().split()])
        self.xyz_data = numpy.reshape(numpy.array([float(i) for i in f.read().split()]), (self.nrows, self.ncols))
        return self.xyz_data


    def _load_release_area_rel(self, infile):
        f = open(infile, "r")
        data = numpy.array([float(i) for i in f.read().split()])
        self.release_area = numpy.reshape(data[1:], (int(data[0]), 2))
        return self.release_area


    def _load_out_hazard_map_asc(self, infile):
        f = open(infile, "r")
        data = numpy.array([float(i) for i in f.read().split()])
        self.hazard_map = numpy.reshape(data, (data.shape[0] / 3, 3))
        return self.hazard_map


    def _load_out_maxheight_asc(self, infile):
        f = open(infile, "r")
        self.max_height = numpy.array([float(i) for i in f.read().split()])
        return self.max_height


    def _load_out_maxvelocity_asc(self, infile):
        f = open(infile, "r")
        self.max_velocity = numpy.array([float(i) for i in f.read().split()])
        return self.max_velocity


    def _load_domain_dom(self, infile):
        f = open(infile, "r")
        self.domain = numpy.array([float(i) for i in f.read().split()])
        return self.domain


    def _load_npz(self, infile):
        files = numpy.load(infile)
        self.absolute_topo = files['arr_0']
        self.relative_topo = files['arr_1']
        return self.absolute_topo, self.relative_topo


    def _load_vertical_npy(self, infile):
        self.vertical_flow = numpy.load(infile)
        self.counter = self.vertical_flow.shape[2] - 1
        return self.vertical_flow


    def _load_horizontal_npy(self, infile):
        self.horizontal_flow = numpy.load(infile)
        self.counter = self.horizontal_flow.shape[2] - 1
        return self.horizontal_flow