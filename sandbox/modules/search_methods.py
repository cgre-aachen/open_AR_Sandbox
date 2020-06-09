import numpy
import matplotlib.pyplot as plt
import matplotlib
import scipy
from scipy.stats import multivariate_normal
import random
from time import sleep
import seaborn as sns

from .module_main_thread import Module


class SearchMethodsModule(Module):
    """
    Module for visualization of different search techniques based on gradients and
    #TODO: have deterministic and probabilistic search methods and later combine them
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, contours=True,
                         cmap='gist_earth_r',
                         over='k',
                         under='k',
                         contours_label=False,
                         minor_contours=True,
                         **kwargs)

        self.xy = (20, 20)
        self.method = 1
        self.show_frame = True
        self.search_active = False
        self.sleep = 0.2
        self.plot_contour_xy = False
        self.plot_xy = True
        self.x = None
        self.y = None
        self.x_list = []
        self.y_list = []
        self.mesh = None
        self.ymax, self.xmax = (self.calib.s_frame_height, self.calib.s_frame_width)
        self.margins_crop = 20
        self.step_variance = 180
        self.init_search = (100, 100)

        self.number_samples = 5000
        self.burn_in_period = 100
        self.counter_t = 0
        self.start_plotting = False
        self.point_color = "red"
        self.marker = "*"
        self.linestyle = "None"
        self.marker_size = 10

        self.activate_frame_capture = True

        self.init_variance = [self.step_variance]

        self.step_search_covariance = 100 # step for generate a new cov
        self.memory_steps = 100  # memory steps, per 100 steps, calculate the accept again

        self.p_sample = [] #ratio of acceptance of samples
        self.counter_total = 0

    def setup(self):
        frame = self.sensor.get_frame()
        if self.crop:
            frame = self.crop_frame(frame)
            frame = self.clip_frame(frame)

        self.plot.render_frame(frame)
        self.projector.frame.object = self.plot.figure

    def update(self):
        # with self.lock:
        #

        if self.activate_frame_capture:
            frame = self.sensor.get_frame()
            if self.crop:
                frame = self.crop_frame(frame)
                frame = self.clip_frame(frame)
                norm_frame = self.calib.s_max - frame
                norm_frame = 1.0/numpy.sum(norm_frame) * norm_frame
                self.create_mesh(norm_frame)
                self.frame = frame

            if self.show_frame:
                self.plot.render_frame(self.frame)
            elif self.show_frame == False:
                self.plot.add_contours(self.frame)
        if self.method != 3 and self.method != 5:
            self.plot.ax.cla()

        # if aruco Module is specified: update, plot aruco markers
        if self.ARUCO_ACTIVE:
            self.update_aruco()
            self.plot.plot_aruco(self.Aruco.aruco_markers)
            self.xy = self.aruco_inside()





        if self.search_active:
            self.plot_search(self.method, self.xy)

        self.projector.trigger() # triggers the update of the bokeh plot

    def create_mesh(self, frame):
        """
        Create a mseh from the topography that can be evaluated in every point.
        Avoid conflicts between pixels(int) and floats
        Args:
            frame: kinect frame

        Returns:

        """
        height_i, width_i = frame.shape
        width = numpy.arange(0, width_i, 1)
        height = numpy.arange(0, height_i, 1)
        xx, yy = numpy.meshgrid(width, height)

        points = numpy.vstack([xx.ravel(), yy.ravel()]).T
        values = frame.ravel()

        self.mesh = scipy.interpolate.LinearNDInterpolator(points, values)
        return True

    def plot_search(self, method, xy):
        """
        selected over the possible options a seacrh algorithm
        Args:
            mesh:
            method:
            xy:

        Returns:

        """
        if method == 1:
            self.activate_frame_capture = True
            self.x_list, self.y_list  = self.gradient_decs_interp(xy, self.mesh)

        elif method == 2:
            self.activate_frame_capture = True
            self.x_list, self.y_list = self.mcmc_random(xy, self.mesh)

        elif method == 3:
            self.activate_frame_capture = False
            if self.x is None and self.y is None:
                self.x_list = []
                self.y_list = []
                if xy is not None:
                    self.x, self.y = xy[0], xy[1]
                else:
                    self.x, self.y = self.init_search

            self.x, self.y = self.mcmc_random_step(self.mesh, self.x, self.y)


        elif method == 4:
            self.activate_frame_capture = True
            self.x_list, self.y_list = self.mcmc_adaptiveMH(self.mesh)

        elif method== 5:
            self.activate_frame_capture = False
            if self.x is None and self.y is None:
                self.x_list = []
                self.y_list = []
                if xy is not None:
                    self.x, self.y = xy[0], xy[1]
                else:
                    self.x, self.y = self.init_search

            self.x, self.y = self.mcmc_adaptiveMH_step(self.mesh, self.x, self.y)

        else:
            return True

        if self.plot_xy:
            self.plot.ax.plot(self.x_list,
                              self.y_list,
                              color = self.point_color,
                              marker = self.marker,
                              markersize = self.marker_size,
                              linestyle = self.linestyle)
        if self.plot_contour_xy:
            sns.kdeplot(self.x_list, self.y_list, ax=self.plot.ax)
            self.plot.ax.get_xaxis().set_visible(False)
            self.plot.ax.get_yaxis().set_visible(False)
            self.plot.ax.set_xlim(0, self.calib.s_frame_width)

            self.plot.ax.set_ylim(0, self.calib.s_frame_height)
            self.plot.ax.set_axis_off()

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

    def gradient_descent(self, xy, mesh):
        alpha = 0.1
        tol = 1e-15
        x , y = int(xy[0]), int(xy[1])
        #z1 = frame[y][x]
        z1 = mesh(x,y)
        frame = 1 # TODO
        x_list = [x]
        y_list = [y]
        deriv1, deriv2 = numpy.gradient(frame)
        for i in range(self.number_samples):
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

    def mcmc_random(self, xy, mesh):
        z = mesh
        if xy is not None:
            x, y = xy[0], xy[1]
        else:
            x, y = self.init_search

        ax = []
        ay = []

        for t in range(0, self.burn_in_period + self.number_samples):
            x_1, y_1 = numpy.random.multivariate_normal(mean=[x, y], cov=[[self.step_variance, 0], [0, self.step_variance]])
            if self.margins_crop < x_1 < self.xmax - self.margins_crop and self.margins_crop < y_1 < self.ymax - self.margins_crop:
                x_sample = x_1
                y_sample = y_1
                accept_prob = min(1, z(x_sample,y_sample) / z(x, y) * 1.0)

                u = random.uniform(0, 1)

                if u < accept_prob:  #
                    x = x_sample
                    y = y_sample

                if t >= self.burn_in_period:
                    ax.append(x)
                    ay.append(y)

        return ax, ay

    def mcmc_random_step(self, mesh, x, y):

        x_1, y_1 = numpy.random.multivariate_normal(mean=[x, y],
                                                    cov=[[self.step_variance, 0],
                                                         [0, self.step_variance]])

        if self.margins_crop < x_1 < self.xmax - self.margins_crop and \
                self.margins_crop < y_1 < self.ymax - self.margins_crop:
            x_sample = x_1
            y_sample = y_1
            accept_prob = min(1, mesh(x_sample, y_sample) / mesh(x,y) * 1.0)
            u = random.uniform(0, 1)

            if self.start_plotting:
                #point = self.plot.ax.plot(x, y, "r.")
                circle1 = plt.Circle((x, y), numpy.sqrt(self.step_variance), fill=False, linewidth = 5)
                circle2 = plt.Circle((x, y), numpy.sqrt(self.step_variance)*2, fill=False, linewidth = 5)
                self.plot.ax.add_artist(circle1)
                self.plot.ax.add_artist(circle2)
                arrow = self.plot.ax.arrow(x, y,
                                           (x_sample - x),
                                           (y_sample - y),
                                           length_includes_head=True,
                                           width=1,
                                           color="black")
                self.projector.trigger()
                plt.pause(self.sleep)

            if u < accept_prob:  #
                x = x_sample
                y = y_sample
                # gren#
                if self.start_plotting:
                    arrow.set_color("green")
                    self.projector.trigger()
                    plt.pause(self.sleep)
            else:
                if self.start_plotting:
                    arrow.set_color("red")
                    self.projector.trigger()
                    plt.pause(self.sleep)

            if self.start_plotting:
                circle1.remove()
                circle2.remove()
                arrow.remove()

            if self.counter_t >= self.burn_in_period:
                self.start_plotting = True
                self.x_list.append(x)
                self.y_list.append(y)

        self.counter_t += 1

        return x, y

    def mcmc_adaptiveMH(self, mesh):
        x, y = (100,100)
        z = mesh
        m = 400  # burn in period
        #n = 5000  # sample numbers, and n/L must be an int
        K_s = 100  # the original cov
        L = 100  # memory steps, per 100 steps, calculate the accept again
        a = 0.23  # resable accept_prob
        epil = 0.01
        K = []
        K.append(K_s)
        step = 100  # step for generate a new cov

        bx_sample = []
        by_sample = []
        P_sample = []  # for the number of accpet probi

        for t in range(0, int(self.number_samples / L)):
            count = 0  # count how many samples have been accept

            if t == 0:
                i = 0
            if t > 0:
                i = m

            while i < L + m:
                x_1, y_1 = numpy.random.multivariate_normal(mean=[x, y], cov=[[K[t], 0], [0, 100]])
                #print(x_1)
                if self.margins_crop < x_1 < self.xmax - self.margins_crop and self.margins_crop < y_1 < self.ymax - self.margins_crop:
                    x_sample = x_1
                    y_sample = y_1
                    #print(x_sample)

                    accept_prob = min(1, z(x_sample,y_sample) / z(x,y) * 1.0)
                    # print('accept_prob:',accept_prob)
                    u = random.uniform(0, 1)
                    # print('u:',u)
                    # ax_sample.append(x_sample)
                    # ay_sample.append(y_sample)

                    if t > 0:  # no burn in period

                        if u < accept_prob:  #
                            x = x_sample
                            y = y_sample
                            count = count + 1

                        bx_sample.append(x)
                        by_sample.append(y)


                    if t == 0:  # only the first time has burn in period

                        if u < accept_prob:  #
                            x = x_sample
                            y = y_sample

                            if i >= m:
                                count = count + 1  # after burn in period, the number of accpetance

                        if i >= m:
                            bx_sample.append(x)
                            by_sample.append(y)

                    i = i + 1


            P_accept = count * 1.0 / L
            P_sample.append(P_accept)
            tau = len(P_sample)
            K.append(K[t])
            if a - epil < P_accept < a + epil:
                K[t + 1] = K[t]
            else:
                if tau >= 2 and numpy.abs(a - P_sample[tau - 1]) > numpy.abs(a - P_sample[tau - 2]):
                    K[tau - 1] = K[tau - 2]
                K[tau] = numpy.abs(numpy.random.normal(K[tau - 1], step))

        return bx_sample, by_sample

    def mcmc_adaptiveMH_step(self, mesh, x, y):
        a = 0.23  # resable accept_prob
        epil = 0.01

        x_1, y_1 = numpy.random.multivariate_normal(mean=[x, y],
                                                    cov=[[self.init_variance[-1], 0], [0, 100]])
        if self.margins_crop < x_1 < self.xmax - self.margins_crop and self.margins_crop < y_1 < self.ymax - self.margins_crop:
            x_sample = x_1
            y_sample = y_1
            accept_prob = min(1, mesh(x_sample, y_sample) / mesh(x, y) * 1.0)

            u = random.uniform(0, 1)

            if self.counter_t == 0:
                if u < accept_prob:  #
                    x = x_sample
                    y = y_sample

                if self.counter_total >= self.burn_in_period:
                    self.x_list.append(x)
                    self.y_list.append(y)
                    self.counter_t += 1
                    self.counter_total = 1

                self.counter_total += 1

            if self.counter_t > 0:
                self.start_plotting = True
                if self.start_plotting:
                    ellipse1 = matplotlib.patches.Ellipse((x,y),
                                                         numpy.sqrt(self.init_variance[-1]),
                                                         numpy.sqrt(100),
                                                         fill = False,
                                                         linewidth = 5)
                    ellipse2 = matplotlib.patches.Ellipse((x, y),
                                                         2*numpy.sqrt(self.init_variance[-1]),
                                                         2*numpy.sqrt(100),
                                                         fill=False,
                                                         linewidth=5)
                    self.plot.ax.add_artist(ellipse1)
                    self.plot.ax.add_artist(ellipse2)
                    arrow = self.plot.ax.arrow(x, y,
                                               (x_sample - x),
                                               (y_sample - y),
                                               length_includes_head=True,
                                               width=1,
                                               color="black")
                    self.projector.trigger()
                    plt.pause(self.sleep)

                if u < accept_prob:  #
                    x = x_sample
                    y = y_sample
                    self.x_list.append(x)
                    self.y_list.append(y)
                    self.counter_total += 1
                    if self.start_plotting:
                        arrow.set_color("green")
                        self.projector.trigger()
                        plt.pause(self.sleep)
                else:
                    if self.start_plotting:
                        arrow.set_color("red")
                        self.projector.trigger()
                        plt.pause(self.sleep)

                self.counter_t += 1

            if self.start_plotting:
                ellipse1.remove()
                ellipse2.remove()
                arrow.remove()

            if self.counter_t == self.memory_steps:
                p_accept = self.counter_total * 1.0 / self.memory_steps
                self.p_sample.append(p_accept)
                tau = len(self.p_sample)

                if a - epil < p_accept < a + epil:
                    self.init_variance.append(self.init_variance[-1])
                    self.counter_t = 1
                    self.counter_total = 1
                else:
                    # TODO: check if is closer to a
                    # making sure the  is better than the previous one
                    if tau >= 2 and numpy.abs(a - self.p_sample[tau - 1]) > numpy.abs(a - self.p_sample[tau - 2]):
                        self.init_variance[-1] = self.init_variance[-2]

                    self.init_variance.append(numpy.abs(numpy.random.normal(self.init_variance[-1], self.step_search_covariance)))
                    self.counter_t = 1
                    self.counter_total = 1

        return x, y



