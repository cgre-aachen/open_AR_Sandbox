import numpy
from scipy.spatial.distance import cdist  # for DummySensor
from scipy.interpolate import griddata  # for DummySensor
from sandbox import set_logger
logger = set_logger(__name__)


class DummySensor:

    def __init__(self, width=512, height=424, depth_limits=(0, 400), extent=None,
                 corners=True, points_n=4, points_distance=0.3,
                 alteration_strength=0.1, **kwargs):
        """

        Args:
            *args:
            extent: [0, width_frame, 0, height_frame, vmin_frame, vmax_frmae]
            corners:
            points_n:
            points_distance:
            alteration_strength:
            **kwargs:
               - random_seed

        """

        random_seed = kwargs.get('random_seed', 1234)
        self.seed = random_seed
        numpy.random.seed(seed=self.seed)

        self.name = 'dummy'
        self.depth_width = width
        self.depth_height = height
        if extent is None:
            self.depth_lim = depth_limits
            self._depth_width = width
            self._depth_height = height
        else:
            self._depth_width = extent[1]
            self._depth_height = extent[3]
            self.depth_lim = extent[-2:]

        self.corners = corners
        self.n = points_n
        # distance in percent of grid diagonal
        self.distance = numpy.sqrt(self._depth_width ** 2 + self._depth_height ** 2) * points_distance
        # alteration_strength: 0 to 1 (maximum 1 equals numpy.pi/2 on depth range)
        self.strength = alteration_strength

        self.grid = None
        self.positions = None
        self.os_values = None
        self.values = None

        # create grid, init values, and init interpolation
        self._create_grid()
        self._pick_positions()
        self._pick_values()
        self._interpolate()
        logger.info("DummySensor initialized.")

    def get_frame(self):
        """

        Returns:

        """
        self._alter_values()
        self._interpolate()
        self.depth[self.depth < 0] = 0  # TODO: Solve the problem of having negative values
        return self.depth

    def _oscillating_depth(self, random):
        r = (self.depth_lim[1] - self.depth_lim[0]) / 2
        return numpy.sin(random) * r + r + self.depth_lim[0]

    def _create_grid(self):
        # creates 2D grid for given resolution
        x, y = numpy.meshgrid(numpy.arange(0, self._depth_width, 1), numpy.arange(0, self._depth_height, 1))
        self.grid = numpy.stack((x.ravel(), y.ravel())).T
        return True

    def _pick_positions(self):
        """
        Param:
            grid: Set of possible points to pick from
            n: desired number of points (without corners counting), not guaranteed to be reached
            distance: distance or range between points
        :return:
        """

        numpy.random.seed(seed=self.seed)
        gl = self.grid.shape[0]
        gw = self.grid.shape[1]
        n = self.n

        if self.corners:
            n += 4
            points = numpy.zeros((n, gw))
            points[1, 0] = self.grid[:, 0].max()
            points[2, 1] = self.grid[:, 1].max()
            points[3, 0] = self.grid[:, 0].max()
            points[3, 1] = self.grid[:, 1].max()
            i = 4  # counter
        else:
            points = numpy.zeros((n, gw))
            # randomly pick initial point
            ipos = numpy.random.randint(0, gl)
            points[0, :2] = self.grid[ipos, :2]
            i = 1  # counter

        while i < n:
            # calculate all distances between remaining candidates and sim points
            dist = cdist(points[:i, :2], self.grid[:, :2])
            # choose candidates which are out of range
            mm = numpy.min(dist, axis=0)
            candidates = self.grid[mm > self.distance]
            # count candidates
            cl = candidates.shape[0]
            if cl < 1:
                break
            # randomly pick candidate and set next point
            pos = numpy.random.randint(0, cl)
            points[i, :2] = candidates[pos, :2]

            i += 1

        # just return valid points if early break occured
        self.positions = points[:i]

        return True

    def _pick_values(self):
        n = self.positions.shape[0]
        self.os_values = numpy.random.uniform(-numpy.pi, numpy.pi, n)
        self.values = self._oscillating_depth(self.os_values)

    def _alter_values(self):
        # maximum range in both directions the values should be altered

        os_range = self.strength * (numpy.pi / 2)
        for i, value in enumerate(self.os_values):
            self.os_values[i] = value + numpy.random.uniform(-os_range, os_range)
        self.values = self._oscillating_depth(self.os_values)

    def _interpolate(self):
        inter = griddata(self.positions[:, :2], self.values, self.grid[:, :2], method='cubic', fill_value=0)
        self.depth = inter.reshape(self._depth_height, self._depth_width)
