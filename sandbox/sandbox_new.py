import json
import numpy as np
import param
import pandas as pd
import panel as pn
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


class Sandbox:
    # TODO Marco: Object as argument required?

    def __init__(self, calibration_file=None):
        self.calibration = Calibration(file=calibration_file)
        #self.projector = Projector(calibration)
        #self.sensor = Sensor(calibration) --> Adjust for correct child class
        #self.model = DummyModel()

class Calibration:
    # TODO Marco: Object as argument required?

    def __init__(self, file=None):
        self.p_width = 800
        self.p_height = 600
        self.p_dpi = 100
        self.p_top_margin = 0
        self.p_left_margin = 0
        self.p_map_width = 400
        self.p_map_height = 200
        # TODO Daniel: Add Kinect calibration parameters

        if file is not None:
            self.load_json(file)

    def load_json(self, file='calibration.json'):
        with open(file) as calibration_json:
            self.__dict__ = json.load(calibration_json)
        print("JSON configuration loaded.")

    def save_json(self, file='calibration.json'):
        with open(file, "w") as calibration_json:
            json.dump(self.__dict__, calibration_json)
        print('JSON configuration file saved:', str(file))

class Projector:

    def __init__(self, calibration, model):
        self.calib = calibration
        self.model = model



    # adjust global panel css
    # the standard deployed bokeh server uses a margin of 8px, remove that margin
    # css = '''
    # body {
    #   margin:0px;
    #   background-color: #ffffff;
    # }
    # .bk.plt_pane {
    # }
    # .panel {
    #   background-color: coral;
    # }
    # '''
    # pn.extension(raw_css=[css])

    def create_panel(self):
        # Create a panel object and serve it within an external bokeh browser that will be opened in a separate window
        # in this special case, a "tight" layout would actually add again white space to the plt canvas, which was already cropped by specifying limits to the axis
        self.mpl_pane = pn.pane.Matplotlib(self.model, width=self.calib.p_map_width , height=self.calib.p_map_height, margin=[0, 0, 0, 0], tight=False,
                                  dpi=self.calib.p_dpi, css_classes=['plt_pane'])

    #self.p_top_margin = 0
    #self.p_left_margin = 0





    # frequency  = pn.widgets.FloatSlider(name='Parameter A', value=2.5, start=0, end=5)
    # amplitude  = pn.widgets.FloatSlider(name='Parameter B', value=5, start=0, end=10)
    # legend     = pn.Column("<br>\n# Legend", frequency, amplitude)
    #
    # # Combine panel and deploy bokeh server
    # bg_panel = pn.Row(mpl_pane, legend, width=pp_width, height=pp_height, sizing_mode='fixed', css_classes=['panel'])
    # # TODO: Add specific port? port=4242
    # bg_panel.show(threaded=False);

#class Sensor:


class Kinect2:
    # will be child class of Sensor

    def __init__(self, calibration):
        self.calib = calibration

    # TODO: Daniel: access and update calibration data program wide with self.calib.s_width = 451 or similar

#class Model:

class DummyModel:
    # will be child class of Model

    def __init__(self, width, height, depth=200, dpi=100):
        self.width = width
        self.height = height
        self.depth = depth
        self.resolution = [width, height, depth]
        self.dpi = dpi

        self.step = 1
        self.points_n = 4
        self.distance = 5

        self.x, self.y, self.z, self.p = self.create_topography(self.resolution, step=self.step, n=self.points_n, distance=self.distance)
        self.figure = self.plot(self.x, self.y, self.z, points=self.p,
                                width=self.width, height=self.height, dpi=self.dpi)

    def update_topo(self):
        self.x, self.y, self.z, self.p = self.create_topography((self.width,self.height,self.depth),
                                                                self.step, n=self.points_n, distance=self.distance)
        self.figure = self.plot(self.x, self.y, self.z, points=self.p,
                                width=self.width, height=self.height, dpi=self.dpi)

    def redraw_plot(self):
        self.figure = self.plot(self.x, self.y, self.z, points=self.p,
                                width=self.width, height=self.height, dpi=self.dpi)

    def pick_points(self, grid, n, distance, seed=None):
        '''
        grid: Set of possible pilot points, can be the grid
        n: desired number of pilot points, not guaranteed to be reached
        distance: distance or range, pilot points should be away from dat points'''

        np.random.seed(seed=seed)

        gl = grid.shape[0]
        gw = grid.shape[1]
        pilots = np.zeros((n, gw))

        # randomly pick initial point
        ipos = np.random.randint(0, gl)
        pilots[0, :2] = grid[ipos, :2]

        i = 1  # counter
        while i < n:

            # calculate all distances between remaining candidates and sim points
            dist = cdist(pilots[:i, :2], grid[:, :2])
            # choose candidates which are out of range
            mm = np.min(dist, axis=0)
            candidates = grid[mm > distance]
            # count candidates
            cl = candidates.shape[0]
            if cl < 1: break
            # randomly pick candidate and set next pilot point
            pos = np.random.randint(0, cl)
            pilots[i, :2] = candidates[pos, :2]

            i += 1

        # just return valid points if early break occured
        pilots = pilots[:i]

        return pilots


    def create_topography(self, resolution, step=1, n=5, distance=3):

        from scipy.interpolate import griddata

        grid_x, grid_y = np.meshgrid(np.arange(0, resolution[0] + step, step), np.arange(0, resolution[1] + step, step))

        # combine position grids for random picking and interpolation
        grid = np.stack((grid_x.ravel(), grid_y.ravel())).T

        # picking positions
        points = self.pick_points(grid, n=n, distance=distance)
        # adding corners
        corners = np.array([[grid_x.min(), grid_y.min()], [grid_x.min(), grid_y.max()], [grid_x.max(), grid_y.min()],
                            [grid_x.max(), grid_y.max()]])
        points = np.vstack([points, corners])
        # picking values
        values = np.random.randint(0, resolution[2], size=n + 4)

        # interpolation
        z = griddata(points, values, grid, method='cubic', fill_value=0)

        # regridding the value grid
        grid_z = np.nan * np.empty((resolution[1], resolution[0]))
        grid_z[grid[:, 1] - 1, grid[:, 0] - 1] = z

        return grid_x, grid_y, grid_z, points


    def plot(self, grid_x, grid_y, grid_z, points=None, cmap='gist_earth', width=800, height=600, dpi=62):

        # z_min, z_max = -np.abs(grid_z).max(), np.abs(grid_z).max()
        ratio = grid_x.max() / grid_y.max()

        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)  # , frameon=False)
        ax = fig.add_axes([0., 0., 1., 1.])

        c = ax.pcolormesh(grid_x, grid_y, grid_z, cmap='gist_earth')
        co = ax.contour(grid_x[:-1, :-1], grid_y[:-1, :-1], grid_z, colors='k')

        if points is not None:
            ax.scatter(points[:, 0], points[:, 1], color='k', marker='x')

        # 1) make patch invisible / like 'frameon=False', buggy
        # fig.patch.set_visible(False)
        # 2) Set whitespace to 0
        # fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
        # 3) Turn off axes and set axes limits
        # ax.axis('off')
        ax.set_axis_off()
        ax.axis([grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])

        # fig.colorbar(c, ax=ax)

        # plt.savefig('test.png', bbox_inches='tight',
        #           transparent=True,
        #           pad_inches=0)

        return fig


#class BlockModel:
    # child class of Model

#class TopoModel:
    # child class of Model

#class GemPyModel:
    # child class of Model