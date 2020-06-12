import numpy
import matplotlib.pyplot as plt
import matplotlib
import scipy
from scipy.stats import multivariate_normal
import random
import seaborn as sns
import panel as pn

from sandbox.modules.module_main_thread import Module


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

        self.speed_alpha = 2
        self.tolerance = 1e-10

        self.xy = (100, 100)
        self.method = 'None'
        self.show_frame = True
        self.search_active = False
        self.sleep = 0.4
        self.plot_contour_xy = False
        self.plot_xy = True
        self.x = None
        self.y = None
        self.x_list = []
        self.y_list = []
        self.mesh = None
        self.mesh_dx = None
        self.mesh_dy = None
        self.ymax, self.xmax = (self.calib.s_frame_height, self.calib.s_frame_width)
        self.margins_crop = 20
        self.step_variance = 180
        self.init_search = (100, 100)
        self.x_grad = []
        self.y_grad = []

        self.number_samples = 5000
        self.burn_in_period = 50
        self.counter_t = 0
        self.start_plotting = False
        self.point_color = "red"
        self.marker = "*"
        self.linestyle = "None"
        self.marker_size = 10
        self.direction_search = "Up"
        self.xy_aruco = []
        self.bins=100

        self.activate_frame_capture = True

        #self.init_variance = [self.step_variance]
        self.init_variance = [150]
        self.step_search_covariance = 100 # step for generate a new cov
        self.memory_steps = 100  # memory steps, per 100 steps, calculate the accept again

        self.p_sample = [] #ratio of acceptance of samples
        self.counter_total = 0
        self.active_gradient_descend = True

        self.hamiltonian = True
        self.hm_frame = None
        self.mesh_hm = None
        self.mesh_dx_hm = None
        self.mesh_dy_hm = None
        self.leapfrog_step = 0.08
        self.leapfrog_points = 30

        self.options = ['None',
                        'Random walk',
                        'Random walk step',
                        'Adaptive HM',
                        'Adaptive HM step',
                        'Hamiltonian MC',
                        'Hamiltonian MC step']

    def setup(self):
        frame = self.sensor.get_frame()
        if self.crop:
            frame = self.crop_frame(frame)
            frame = self.clip_frame(frame)

        self.plot.render_frame(frame)
        self.projector.frame.object = self.plot.figure
        self._create_widgets()

    def update(self):
        # with self.lock:
        if self.activate_frame_capture:
            frame = self.sensor.get_frame()
            if self.crop:
                frame = self.crop_frame(frame)
                frame = self.clip_frame(frame)
                norm_frame = self.calib.s_max - frame
                norm_frame = 1.0/numpy.sum(norm_frame) * norm_frame
                self.mesh = self.create_mesh(norm_frame)
                self.frame = frame
            if self.active_gradient_descend:
                der_y, der_x = numpy.gradient(self.frame)
                self.mesh_dx = self.create_mesh(der_x)
                self.mesh_dy = self.create_mesh(der_y)

            if self.hamiltonian:
                self.hm_frame = - numpy.log(norm_frame)
                self.mesh_hm = self.create_mesh(self.hm_frame)
                der_y_hm, der_x_hm = numpy.gradient(self.hm_frame)
                self.mesh_dx_hm = self.create_mesh(der_x_hm, fill_value=0.0)
                self.mesh_dy_hm = self.create_mesh(der_y_hm, fill_value=-0.0)

            self.plot.render_frame(self.frame)

        else:
            if self.show_frame:
                self.plot.render_frame(self.frame)
            else:
                self.plot.add_contours(self.frame)

        # if aruco Module is specified: update, plot aruco markers
        if self.ARUCO_ACTIVE:
            self.update_aruco()
            self.plot.plot_aruco(self.Aruco.aruco_markers)
            self.xy_aruco = self.aruco_inside()

        self.x_grad, self.y_grad = self.gradient_descent(self.xy_aruco, self.mesh, self.mesh_dx, self.mesh_dy, self.direction_search)
        for i in range(len(self.x_grad)):
            self.plot.ax.plot(self.x_grad[i], self.y_grad[i], "r*-")
            self.plot.ax.plot(self.x_grad[i][-1], self.y_grad[i][-1], "b*", markersize=20)

        if self.search_active:
            self.plot_search(self.method, self.xy)

        self.projector.trigger() # triggers the update of the bokeh plot

    def create_mesh(self, frame, fill_value=numpy.nan):
        """
        Create a mseh from the topography that can be evaluated in every point.
        Avoid conflicts between pixels(int) and floats
        Args:
            frame: kinect frame
            fill_value:

        Returns:

        """
        height_i, width_i = frame.shape
        width = numpy.arange(0, width_i)
        height = numpy.arange(0, height_i)
        xx, yy = numpy.meshgrid(width, height)

        points = numpy.vstack([xx.ravel(), yy.ravel()]).T
        values = frame.ravel()

        mesh = scipy.interpolate.LinearNDInterpolator(points, values, fill_value=fill_value)

        return mesh

    def plot_search(self, method, xy):
        """
        selected over the possible options a seacrh algorithm
        Args:
            mesh:
            method:
            xy:

        Returns:

        """
        if method == self.options[1]:
            self.activate_frame_capture = True
            self.x_list = []
            self.y_list = []
            self.x_list, self.y_list = self.mcmc_random(xy, self.mesh)

        elif method == self.options[2]:
            self.activate_frame_capture = False
            if self.x is None and self.y is None:
                self.x_list = []
                self.y_list = []
                if xy is not None:
                    self.x, self.y = xy[0], xy[1]
                else:
                    self.x, self.y = self.init_search

            self.x, self.y = self.mcmc_random_step(self.mesh, self.x, self.y)

        elif method == self.options[3]:
            self.activate_frame_capture = True
            self.x_list = []
            self.y_list = []
            self.x_list, self.y_list = self.mcmc_adaptiveMH(self.mesh)

        elif method == self.options[4]:
            self.activate_frame_capture = False
            if self.x is None and self.y is None:
                self.x_list = []
                self.y_list = []
                if xy is not None:
                    self.x, self.y = xy[0], xy[1]
                else:
                    self.x, self.y = self.init_search

            self.x, self.y = self.mcmc_adaptiveMH_step(self.mesh, self.x, self.y)

        elif method == self.options[5]:
            self.activate_frame_capture = True
            #self.x_list = []
            #self.y_list = []
            self.x_list, self.y_list = self.mcmc_hamiltonianMC(self.mesh_hm,
                                                               self.mesh_dx_hm,
                                                               self.mesh_dy_hm)
        elif method == self.options[6]:
            self.activate_frame_capture = False
            if self.x is None and self.y is None:
                self.x_list = []
                self.y_list = []
                if xy is not None:
                    self.x, self.y = xy[0], xy[1]
                else:
                    self.x, self.y = self.init_search

            self.x, self.y = self.mcmc_hamiltonianMC_step(self.mesh_hm,
                                                          self.mesh_dx_hm,
                                                          self.mesh_dy_hm,
                                                          self.x,
                                                          self.y)
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
        xy = []
        if len(df_position) > 0:
            xy = df_position.loc[df_position.is_inside_box == True, ('box_x', 'box_y')].values
        return xy

    def gradient_descent(self, xy, mesh, dx, dy, direction):
        x_sol = []
        y_sol = []
        for i in range(len(xy)):
            x, y = xy[i]
            z1 = mesh(x, y)
            z2 = 0
            x_list = [x]
            y_list = [y]
            count = 0
            while numpy.abs(z1 - z2) > self.tolerance and count <= self.number_samples:
                z1 = mesh(x, y)
                if direction == "Down":
                    x = (x + self.speed_alpha * dx(x, y))
                    y = (y + self.speed_alpha * dy(x, y))
                else:
                    x = (x - self.speed_alpha * dx(x, y))
                    y = (y - self.speed_alpha * dy(x, y))

                z2 = mesh(x, y)
                x_list.append(x)
                y_list.append(y)
                count += 1
            x_sol.append(x_list)
            y_sol.append(y_list)

        return x_sol, y_sol

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
        m = 50  # burn in period
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
                                                    cov=[[self.init_variance[-1], 0], [0, self.step_variance]])
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
                                                         numpy.sqrt(self.step_variance),
                                                         fill = False,
                                                         linewidth = 5)
                    ellipse2 = matplotlib.patches.Ellipse((x, y),
                                                         2*numpy.sqrt(self.init_variance[-1]),
                                                         2*numpy.sqrt(self.step_variance),
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

    def mcmc_hamiltonianMC(self, mesh, dx_mesh, dy_mesh, n = 700, l = 10):
        x, y = self.init_search
        x_list = [x]
        y_list = [y]
        #for t in range(self.number_samples):
        for t in range(n):
            # sample random momentum
            proposed_momentum_vector = numpy.random.multivariate_normal(mean=[0, 0],
                                                        cov=[[self.step_variance, 0], [0, self.step_variance]])
           # proposed_momentum_vector = numpy.random.normal(0, self.step_variance, 2)
            # leapfrog
            # if the xStar[,] is inside the frame or not
            #if self.margins_crop < x < self.xmax - self.margins_crop and self.margins_crop < y < self.ymax - self.margins_crop:
                # gradient
            gradient_xy = numpy.array([dx_mesh(x, y), dy_mesh(x, y)])
            # momentum, make a half step for momentum at the beginning
            momentum_vector = proposed_momentum_vector - (self.leapfrog_step * gradient_xy) / 2.0
            # lepfrog steps
            x_sample, y_sample = x, y
            #for j in range(self.leapfrog_points - 1):
            for j in range(l):
                # postion/sample,full step for position/sample
                x_sample, y_sample = numpy.array([x_sample, y_sample]) + self.leapfrog_step * momentum_vector
                #                 #if 0 <= xStar[0] < xmax - 2 and 0 <= xStar[1] < ymax - 2:
                #                 # find new gradient
                gradient_xy = numpy.array([dx_mesh(x_sample, y_sample), dy_mesh(x_sample, y_sample)])
                # momentum,full step of momentum
                momentum_vector = momentum_vector - self.leapfrog_step * gradient_xy
            # last half step
            x_sample, y_sample = numpy.array([x_sample, y_sample]) + self.leapfrog_step * momentum_vector
            #if 0 <= xStar[0] < xmax - 2 and 0 <= xStar[1] < ymax - 2:
            gradient_xy = numpy.array([dx_mesh(x_sample, y_sample), dy_mesh(x_sample, y_sample)])
            momentum_vector = momentum_vector - (self.leapfrog_step * gradient_xy) / 2.0

            # evaluate energy
            #if 0 <= xStar[0] < xmax - 2 and 0 <= xStar[1] < ymax - 2:
            potential_energy = mesh(x_sample, y_sample)
            previous_potential_energy = mesh(x, y)

            #### Kinetic energy function  KE = (p @ p) / 2.0
            previous_kinetic_e = numpy.vdot(proposed_momentum_vector, proposed_momentum_vector) / 2.0
            kinetic_e = numpy.vdot(momentum_vector, momentum_vector) / 2.0

            # acceptance
            accept_prob = min(1, numpy.exp(- potential_energy + previous_potential_energy -
                                     kinetic_e + previous_kinetic_e))
            uc = numpy.random.rand()

            if uc < accept_prob:
                x = x_sample
                y = y_sample
                x_list.append(x)
                y_list.append(y)

        return x_list, y_list

    def mcmc_hamiltonianMC_step(self, mesh, dx_mesh, dy_mesh, x, y):
        #for t in range(self.number_samples):
        print(x)

        # sample random momentum
        proposed_momentum_vector = numpy.random.multivariate_normal(mean=[0, 0],
                                                    cov=[[self.step_variance, 0], [0, self.step_variance]])
       # proposed_momentum_vector = numpy.random.normal(0, self.step_variance, 2)
        # leapfrog
        # if the xStar[,] is inside the frame or not
        #if self.margins_crop < x < self.xmax - self.margins_crop and self.margins_crop < y < self.ymax - self.margins_crop:
            # gradient
        gradient_xy = numpy.array([dx_mesh(x, y), dy_mesh(x, y)])
        # momentum, make a half step for momentum at the beginning
        momentum_vector = proposed_momentum_vector - (self.leapfrog_step * gradient_xy) / 2.0
        # lepfrog steps
        x_sample, y_sample = x, y
        x_temp = [x]
        y_temp = [y]
        #for j in range(self.leapfrog_points - 1):
        for j in range(self.leapfrog_points):
            # postion/sample,full step for position/sample
            x_sample, y_sample = numpy.array([x_sample, y_sample]) + self.leapfrog_step * momentum_vector
            #                 #if 0 <= xStar[0] < xmax - 2 and 0 <= xStar[1] < ymax - 2:
            #                 # find new gradient
            gradient_xy = numpy.array([dx_mesh(x_sample, y_sample), dy_mesh(x_sample, y_sample)])
            # momentum,full step of momentum
            momentum_vector = momentum_vector - self.leapfrog_step * gradient_xy
            x_temp.append(x_sample)
            y_temp.append(y_sample)
        # last half step
        self.plot.ax.plot(x_temp, y_temp, "k*-")
        self.projector.trigger()
        plt.pause(self.sleep)

        x_sample, y_sample = numpy.array([x_sample, y_sample]) + self.leapfrog_step * momentum_vector
        #if 0 <= xStar[0] < xmax - 2 and 0 <= xStar[1] < ymax - 2:
        gradient_xy = numpy.array([dx_mesh(x_sample, y_sample), dy_mesh(x_sample, y_sample)])
        momentum_vector = momentum_vector - (self.leapfrog_step * gradient_xy) / 2.0

        # evaluate energy
        #if 0 <= xStar[0] < xmax - 2 and 0 <= xStar[1] < ymax - 2:
        potential_energy = mesh(x_sample, y_sample)
        previous_potential_energy = mesh(x, y)

        #### Kinetic energy function  KE = (p @ p) / 2.0
        previous_kinetic_e = numpy.vdot(proposed_momentum_vector, proposed_momentum_vector) / 2.0
        kinetic_e = numpy.vdot(momentum_vector, momentum_vector) / 2.0

        # acceptance
        accept_prob = min(1, numpy.exp(- potential_energy + previous_potential_energy -
                                 kinetic_e + previous_kinetic_e))
        uc = numpy.random.rand()

        if uc < accept_prob:
            x = x_sample
            y = y_sample
            self.x_list.append(x)
            self.y_list.append(y)


        #step_plot.remove()

        return x, y


    def hist_target_distribution(self):
        """fig = plt.figure(figsize=(7, 20))
        ax_x = fig.add_subplot(221)
        ax_x_.hist(bx_sample, bins=100)
        ax_x.set_title('Adaptive MH samples')

        ax_y = fig.add_subplot(222)
        ax_y.hist(ys[:, 0], bins=100)
        ax_y.set_title('Target distribution')
        """
        pass

    def hist_sampled_distribution(self):
        fig = plt.figure(figsize=(7, 10))
        ax_x = fig.add_subplot(121)
        ax_x.hist(self.x_list, bins=self.bins)
        ax_x.set_title('x_direction')

        ax_y = fig.add_subplot(122)
        ax_y.hist((self.y_list), bins=self.bins)
        ax_y.set_title('y direction')
        plt.show()

    # Widgets

    def show_widgets(self):
        widget = pn.WidgetBox(self._widget_search_active,
                              self._widget_plot_points,
                              self._widget_plot_contour,
                              self._widget_show_frame)
        column = pn.Column(self._widget_selector, widget)

        tabs = pn.Tabs(('Controllers', column),
                      ("Plot", self.widget_plot_module()))

        return tabs

    def _create_widgets(self):
        self._widget_selector = pn.widgets.RadioButtonGroup(
            name='Select search method',
            options=self.options,
            value=self.method,
            button_type='success'
        )
        self._widget_selector.param.watch(self._callback_selector, 'value', onlychanged=False)

        self._widget_search_active = pn.widgets.Checkbox(name='Start the search', value=self.search_active)
        self._widget_search_active.param.watch(self._callback_search_active, 'value',
                                                 onlychanged=False)

        self._widget_plot_points = pn.widgets.Checkbox(name='Show the points', value=self.plot_xy)
        self._widget_plot_points.param.watch(self._callback_plot_points, 'value',
                                                onlychanged=False)

        self._widget_plot_contour = pn.widgets.Checkbox(name='Show the contours of the points', value=self.plot_contour_xy)
        self._widget_plot_contour.param.watch(self._callback_plot_contour, 'value',
                                             onlychanged=False)

        self._widget_show_frame = pn.widgets.Checkbox(name='Show the frame', value=self.show_frame)
        self._widget_show_frame.param.watch(self._callback_show_frame, 'value',
                                              onlychanged=False)

    def _callback_selector(self, event):
        self.method = event.new

    def _callback_search_active(self, event):
        self.search_active = event.new

    def _callback_plot_points(self, event):
        self.plot_xy = event.new

    def _callback_plot_contour(self, event):
        self.plot_contour_xy = event.new

    def _callback_show_frame(self, event):
        self.show_frame = event.new
