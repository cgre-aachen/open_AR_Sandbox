import numpy
import matplotlib.pyplot as plt
import scipy
from scipy.stats import multivariate_normal
import random

from .module_main_thread import Module


class SearchMethodsModule(Module):
    """
    Module for visualization of different search techniques based on gradients and geostatistics
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, contours=True,
                         cmap='gist_earth',
                         over='k',
                         under='k',
                         contours_label=True,
                         minor_contours=True,
                         **kwargs)

        self.xy = (20, 20)
        self.method = 1
        self.show_frame = True

    def setup(self):
        frame = self.sensor.get_frame()
        if self.crop:
            frame = self.crop_frame(frame)
            frame = self.clip_frame(frame)

        self.plot.render_frame(frame)
        self.projector.frame.object = self.plot.figure

    def update(self):
        # with self.lock:
        frame = self.sensor.get_frame()
        if self.crop:
            frame = self.crop_frame(frame)
            frame = self.clip_frame(frame)
            norm_frame = self.calib.s_max - frame

        #self.plot.ax.cla()

        # if aruco Module is specified: update, plot aruco markers
        if self.ARUCO_ACTIVE:
            self.update_aruco()
            self.plot.plot_aruco(self.Aruco.aruco_markers)
            self.xy = self.aruco_inside()

        self.plot.ax.cla()
        if self.show_frame:
            self.plot.render_frame(frame)

        self.plot_search(norm_frame, self.method, self.xy)

        self.projector.trigger() # triggers the update of the bokeh plot

    def plot_search(self, frame, method, xy):
        if method == 1:

            x, y = self.gradient_descent(5000, frame, xy)
        elif method ==2:

            x, y = self.mcmc_random(4000, 100, frame, xy)
        else:
            return True

        self.plot.ax.plot(x,y, "r*")

    def aruco_inside(self):
        df_position = self.Aruco.aruco_markers
        xy = None
        if len(df_position) > 0:
            xy = df_position.loc[df_position.is_inside_box == True, ('box_x', 'box_y')]
            if len(xy) > 0:
                xy = xy.values[0]
            else:
                xy =None

        return xy

    def gradient_descent(self, n, frame, xy):
        alpha = 0.1
        tol = 1e-15
        x , y = int(xy[0]), int(xy[1])
        z1 = frame[y][x]
        x_list = [x]
        y_list = [y]
        deriv1, deriv2 = numpy.gradient(frame)
        for i in range(n):
            #dimension x
            if deriv1[y][x] < 0 :
                y_pos = y + 1
            elif deriv1[y][x] > 0:
                y_pos = y -1
            else:
                y_pos = y

            if deriv2[y][x] < 0 :
                x_pos = x + 1
            elif deriv2[y][x] > 0:
                x_pos = x - 1
            else:
                x_pos = x

            if deriv1[y][x] < deriv2[y][x]:
                y = y_pos
            elif deriv1[y][x] > deriv2[y][x]:
                x = x_pos
            else:
                x= x_pos
                y = y_pos

            #z2 = frame[x][y]

            x_list.append(x)
            y_list.append(y)

            #if z1 - z2 < tol :
             #   return x_list, y_list, x, y
            #if z2 < z1:
              #  z1 = z2
        return x_list, y_list

    def gradient_decs_interp(n, frame, xy):
        alpha = 0.005
        tol = 1e-10
        x, y = int(xy[0]), int(xy[1])
        z1 = frame[y][x]
        z2 = 0
        x_list = [x]
        y_list = [y]
        der_y, der_x = numpy.gradient(frame)
        count = 0
        for i in range(n):
            # while np.abs(z1 - z2) < tol or count <= n:

            x_up = int(numpy.ceil(x))
            x_down = int(numpy.floor(x))

            if x_up == x_down:
                x_up += 1

            y_up = int(numpy.ceil(y))
            y_down = int(numpy.floor(y))

            if y_up == y_down:
                y_up += 1

            points = numpy.array([[x_up, y_up],
                               [x_up, y_down],
                               [x_down, y_up],
                               [x_down, y_down]])
            values = numpy.array([frame[y_up][x_up],
                               frame[y_up][x_down],
                               frame[y_down][x_up],
                               frame[y_down][x_down]])
            inter = scipy.interpolate.LinearNDInterpolator(points, values)

            x_ = inter(x_down, y) - inter(x_up, y)
            y_ = inter(x, y_down) - inter(x, y_up)

            dx = x_ / (x_down - x_up)
            dy = y_ / (y_down - y_up)

            z1 = inter(x, y)
            x = (x - alpha * dx)
            y = (y - alpha * dy)
            z2 = inter(x, y)

            x_list.append(x)
            y_list.append(y)
            count += 1
        # print (count)

        return x_list, y_list

    def mcmc_random(self, n, m, frame, xy):
        ymax, xmax = frame.shape
        z = frame
        x, y = int(xy[0]), int(xy[1])
        ax = []
        ay = []
        # ax_sample =[]
        # ay_sample =[]
        crop = 20
        step = 180

        for t in range(0, m + n):
            x_1, y_1 = numpy.random.multivariate_normal(mean=[x, y], cov=[[step, 0], [0, step]])
            if crop < x_1 < xmax - crop and crop < y_1 < ymax - crop:
                x_sample = int(x_1)
                y_sample = int(y_1)
                accept_prob = min(1, z[y_sample][x_sample] / z[y][x] * 1.0)
                # print('accept_prob:',accept_prob)
                u = random.uniform(0, 1)
                # print('u:',u)
                # ax_sample.append(x_sample)
                # ay_sample.append(y_sample)

                if u < accept_prob:  #
                    x = x_sample
                    y = y_sample
                    # ax_sandbox.plot(x, y,'r*' )
                    # print(x,y)
                    # time.sleep(1)
                    # print('x:,y:',x,y)

                if t >= m:
                    ax.append(x)
                    ay.append(y)
                    # ax_sandbox.plot(x, y,'r*' )
                    # time.sleep(1)

        return ax, ay

