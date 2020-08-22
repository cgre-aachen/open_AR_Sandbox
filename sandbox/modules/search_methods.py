import numpy
import matplotlib.pyplot as plt
import matplotlib
import scipy
import random
import seaborn as sns
import panel as pn

from .template import ModuleTemplate


class SearchMethodsModule(ModuleTemplate):
    """
    Module for visualization of different search techniques based on gradients and
    #TODO: have deterministic and probabilistic search methods and later combine them
    """
    def __init__(self, *args, extent, **kwargs):

        self.speed_alpha = 2
        self.tolerance = 1e-10

        self.xy = (100, 100)

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
        self.ymax, self.xmax = (extent[3], extent[1])
        self.margins_crop = 20
        self.step_variance = 400
        self.init_search = (100, 100)
        self.x_grad = []
        self.y_grad = []

        self.number_samples = 5000
        self.burn_in_period = 10
        self.counter_t = 0
        self.start_plotting = False
        self.point_color = "red"
        self.marker = "*"
        self.linestyle = "None"
        self.marker_size = 10
        self.direction_search = "Maximum"
        self.xy_aruco = []
        self.bins=100

        self.activate_frame_capture = True

        self.init_variance = [self.step_variance]
        #self.init_variance = [150]
        self.step_search_covariance = 100 # step for generate a new cov
        self.memory_steps = 10  # memory steps, per 100 steps, calculate the accept again

        self.p_sample = [] #ratio of acceptance of samples
        self.counter_total = 0
        self.active_gradient_descend = True

        self.hamiltonian = True
        self.frame_hm = None
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
        self.method = 'None'

        self.histogram = pn.pane.Matplotlib(plt.figure(), tight=False)#, height=335)
        plt.close()
        self.trigger = None
        self.lock = None

        self._create_widgets()


    def update(self, sb_params: dict):
        self.frame = sb_params.get('frame')
        self.lock = sb_params.get('lock_frame')
        ax = sb_params.get('ax')
        #extent=sb_params.get('extent')
        same_frame = sb_params.get('same_frame')
        self.trigger = sb_params.get('trigger')
        if self.activate_frame_capture:
            if not same_frame:
                self.update_mesh(self.frame, margins_crop=self.margins_crop, fill_value=0)#, extent=extent)
            sb_params['freeze_frame'] = False
        else:
            sb_params['freeze_frame'] = True
        marker = sb_params.get('marker')
        if len(marker) > 0:
            self.xy_aruco = marker.loc[marker.is_inside_box, ('box_x', 'box_y')].values
        else:
            self.xy_aruco=[]

        self.plot(ax)

        return sb_params

    def plot(self, ax):
        self.remove_lines(ax)
        if self.active_gradient_descend:
            self.x_grad, self.y_grad = self.gradient_descent(self.xy_aruco, self.mesh, self.mesh_dx, self.mesh_dy, self.direction_search)
            for i in range(len(self.x_grad)):
                ax.plot(self.x_grad[i], self.y_grad[i], "r*-")
                ax.plot(self.x_grad[i][-1], self.y_grad[i][-1], "b*", markersize=20)

        if self.search_active:
            self.plot_search(self.method, self.xy, ax)

    def remove_lines(self, ax):
        [lines.remove() for lines in reversed(ax.lines) if isinstance(lines, matplotlib.lines.Line2D)]

    def create_mesh(self, frame, fill_value=numpy.nan, margins_crop: int = 0):
        """
        Create a mseh from the topography that can be evaluated in every point.
        Avoid conflicts between pixels(int) and floats
        Args:
            frame: kinect frame
            fill_value:

        Returns:

        """
        if margins_crop!=0:
            frame = frame[margins_crop:-margins_crop, margins_crop:-margins_crop]

        height_i, width_i = frame.shape
        width = numpy.arange(margins_crop, width_i + margins_crop)
        height = numpy.arange(margins_crop, height_i + margins_crop)
        xx, yy = numpy.meshgrid(width, height)

        points = numpy.vstack([xx.ravel(), yy.ravel()]).T
        values = frame.ravel()

        mesh = scipy.interpolate.LinearNDInterpolator(points, values, fill_value=fill_value)

        return mesh

    def update_mesh(self, frame, fill_value=numpy.nan, margins_crop=0):
        """
        create the mesh for the current frame and its derivatives
        Args:
            frame:
            fill_value:
            margins_crop:

        Returns:
        """
        der_y, der_x = numpy.gradient(frame)
        self.mesh_dx = self.create_mesh(der_x, margins_crop=margins_crop, fill_value=fill_value)
        self.mesh_dy = self.create_mesh(der_y, margins_crop=margins_crop, fill_value=fill_value)

        frame = 1.0 / numpy.sum(frame) * frame
        self.mesh = self.create_mesh(frame, margins_crop=margins_crop, fill_value=fill_value)

        hm_frame = (-1) * (numpy.log(frame))
        self.mesh_hm = self.create_mesh(hm_frame, margins_crop=margins_crop, fill_value=fill_value)
        der_y_hm, der_x_hm = numpy.gradient(hm_frame)
        self.mesh_dx_hm = self.create_mesh(der_x_hm, margins_crop=margins_crop, fill_value=fill_value)
        self.mesh_dy_hm = self.create_mesh(der_y_hm, margins_crop=margins_crop, fill_value=fill_value)

        self.frame_norm = frame
        self.frame_hm = hm_frame

        return True

    def plot_search(self, method, xy, ax):
        """
        selected over the possible options a search algorithm
        Args:
            mesh:
            method:
            xy:

        Returns:

        """
        if self.plot_contour_xy:
            ax.collections = [] #TODO: Improve this
            ax = sns.kdeplot(self.x_list, self.y_list, ax=ax)
            self.trigger()

        if self.plot_xy:
            ax.plot(self.x_list,
                              self.y_list,
                              color = self.point_color,
                              marker = self.marker,
                              markersize = self.marker_size,
                              linestyle = self.linestyle)

            #ax.get_xaxis().set_visible(False)
            #ax.get_yaxis().set_visible(False)
            #ax.set_xlim(0, self.calib.s_frame_width)
            #self.plot.ax.set_ylim(0, self.calib.s_frame_height)
            #self.plot.ax.set_axis_off()

        if method == self.options[1]:
            #self.activate_frame_capture = True
            self.x_list = []
            self.y_list = []
            self.x_list, self.y_list = self.mcmc_random(xy, self.mesh)

        elif method == self.options[2]:
            #self.activate_frame_capture = False
            if self.x is None and self.y is None:
                self.x_list = []
                self.y_list = []
                if xy is not None:
                    self.x, self.y = xy[0], xy[1]
                else:
                    self.x, self.y = self.init_search

            self.x, self.y = self.mcmc_random_step(self.mesh, self.x, self.y, ax)

        elif method == self.options[3]:
            #self.activate_frame_capture = True
            self.x_list = []
            self.y_list = []
            self.x_list, self.y_list = self.mcmc_adaptiveMH(self.mesh)

        elif method == self.options[4]:
            #self.activate_frame_capture = False
            if self.x is None and self.y is None:
                self.x_list = []
                self.y_list = []
                if xy is not None:
                    self.x, self.y = xy[0], xy[1]
                else:
                    self.x, self.y = self.init_search

            self.x, self.y = self.mcmc_adaptiveMH_step(self.mesh, self.x, self.y, ax)

        elif method == self.options[5]:
            #self.activate_frame_capture = True
            self.x_list = []
            self.y_list = []
            self.x_list, self.y_list = self.mcmc_hamiltonianMC(self.mesh_hm,
                                                               self.mesh_dx_hm,
                                                               self.mesh_dy_hm)
        elif method == self.options[6]:
            #self.activate_frame_capture = False
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
                                                          self.y, ax)
        else:
            return False

        return True

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
                if direction == "Maximum":
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
        if len(xy) > 0:
            x, y = xy[0], xy[1]
        else:
            x, y = self.init_search

        ax = []
        ay = []

        for t in range(0, self.burn_in_period + self.number_samples):
            x_1, y_1 = numpy.random.multivariate_normal(mean=[x, y], cov=[[self.step_variance, 0], [0, self.step_variance]])
            #if self.margins_crop < x_1 < self.xmax - self.margins_crop and self.margins_crop < y_1 < self.ymax - self.margins_crop:
            x_sample = x_1
            y_sample = y_1
            accept_prob = min(1, mesh(x_sample, y_sample) / mesh(x, y) * 1.0)

            u = random.uniform(0, 1)

            if u < accept_prob:  #
                x = x_sample
                y = y_sample

            if t >= self.burn_in_period:
                ax.append(x)
                ay.append(y)

        return ax, ay

    def mcmc_random_step(self, mesh, x, y, ax):
        x_1, y_1 = numpy.random.multivariate_normal(mean=[x, y],
                                                    cov=[[self.step_variance, 0],
                                                         [0, self.step_variance]])

        #if self.margins_crop < x_1 < self.xmax - self.margins_crop and \
         #       self.margins_crop < y_1 < self.ymax - self.margins_crop:
        x_sample = x_1
        y_sample = y_1
        accept_prob = min(1, mesh(x_sample, y_sample) / mesh(x, y) * 1.0)
        u = random.uniform(0, 1)

        if self.start_plotting:
            #point = self.plot.ax.plot(x, y, "r.")
            circle1 = plt.Circle((x, y), numpy.sqrt(self.step_variance), fill=False, linewidth = 5)
            circle2 = plt.Circle((x, y), numpy.sqrt(self.step_variance)*2, fill=False, linewidth = 5)
            ax.add_artist(circle1)
            ax.add_artist(circle2)
            arrow = ax.arrow(x, y,
                                       (x_sample - x),
                                       (y_sample - y),
                                       length_includes_head=True,
                                       width=1,
                                       color="black")
            self.trigger()
            plt.pause(self.sleep)

        if u < accept_prob:  #
            x = x_sample
            y = y_sample
            # gren#
            if self.start_plotting:
                arrow.set_color("green")
                self.trigger()
                plt.pause(self.sleep)
        else:
            if self.start_plotting:
                arrow.set_color("red")
                self.trigger()
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
                #if self.margins_crop < x_1 < self.xmax - self.margins_crop and self.margins_crop < y_1 < self.ymax - self.margins_crop:
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

    def mcmc_adaptiveMH_step(self, mesh, x, y, ax):
        a = 0.23  # resable accept_prob
        epil = 0.01

        x_1, y_1 = numpy.random.multivariate_normal(mean=[x, y],
                                                    cov=[[self.init_variance[-1], 0], [0, self.step_variance]])
        #if self.margins_crop < x_1 < self.xmax - self.margins_crop and self.margins_crop < y_1 < self.ymax - self.margins_crop:
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
                ax.add_artist(ellipse1)
                ax.add_artist(ellipse2)
                arrow = ax.arrow(x, y,
                                           (x_sample - x),
                                           (y_sample - y),
                                           length_includes_head=True,
                                           width=1,
                                           color="black")
                self.trigger()
                plt.pause(self.sleep)

            if u < accept_prob:  #
                x = x_sample
                y = y_sample
                self.x_list.append(x)
                self.y_list.append(y)
                self.counter_total += 1
                if self.start_plotting:
                    arrow.set_color("green")
                    self.trigger()
                    plt.pause(self.sleep)
            else:
                if self.start_plotting:
                    arrow.set_color("red")
                    self.trigger()
                    plt.pause(self.sleep)

            self.counter_t += 1

        if self.start_plotting:
            ellipse1.remove()
            ellipse2.remove()
            arrow.remove()

        if self.counter_t >= self.memory_steps:
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

            if uc < accept_prob and self.margins_crop < x_sample < self.xmax - self.margins_crop and self.margins_crop < y_sample < self.ymax - self.margins_crop:
                x = x_sample
                y = y_sample
                x_list.append(x)
                y_list.append(y)

        return x_list, y_list

    def mcmc_hamiltonianMC_step(self, mesh, dx_mesh, dy_mesh, x, y, ax):
        #for t in range(self.number_samples):
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
        ax.plot(x_temp, y_temp, "k*-")
        self.trigger()
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

        if uc < accept_prob and self.margins_crop < x_sample < self.xmax - self.margins_crop and self.margins_crop < y_sample < self.ymax - self.margins_crop:
            x = x_sample
            y = y_sample
            self.x_list.append(x)
            self.y_list.append(y)


        #step_plot.remove()

        return x, y

    def hist_distribution(self):
        """
        Histogram showing the distribution of the sampled points and the targer
        Returns:

        """
        #x = self.frame_norm.sum(axis=0)
        #y = self.frame_norm.sum(axis=1)

        fig = plt.figure(figsize=(10, 10))
        ax_x = fig.add_subplot(121)
        ax_x = sns.distplot(self.x_list, bins=self.bins, ax=ax_x)
        #ax_x.hist(self.x_list, bins=self.bins)
        #ax_x.plot(range(len(x)), x, "r--")
        ax_x.set_title('x_direction')

        ax_y = fig.add_subplot(122)
        #ax_y.hist((self.y_list), bins=self.bins)
        ax_x = sns.distplot(self.y_list, bins=self.bins, ax=ax_y)
        #ax_y.plot(range(len(y)), y, "r--")
        ax_y.set_title('y direction')

        return fig

    # Widgets

    def show_widgets(self):
        widget = pn.Column(self._widget_activate_gradient_descend,
                           self._widget_optimum,
                           self._widget_speed_gradient_descend,
                           "<b>Frame controller</b>",
                           self._widget_activate_frame_capture,
                           self._widget_refresh_frame)

        column = pn.Column("#Simulation options",
                           self._widget_selector,
                           self._widget_search_active,
                           "<b>Controllers</b>",
                           self._widget_sleep,
                           self._widget_margins_crop,
                           "<b>Visualization options</b>",
                           self._widget_plot_points,
                           self._widget_plot_contour,
                           "<b>Modify constants</b> ",
                           self._widget_variance,
                           self._widget_number_samples,
                           "<b>Adaptive options</b>",
                           self._widget_memory_steps,
                           "<b> Hamiltonian options",
                           self._widget_leapfrog_step,
                           self._widget_leapfrog_points
                           )
        histogram = pn.WidgetBox(self._widget_histogram_refresh,
                                 self.histogram)
        row = pn.Row(column, widget)

        tabs = pn.Tabs(('Controllers', row),
                       ("Histogram plot", histogram),
                       )

        return tabs

    def _create_widgets(self):
        self._widget_selector = pn.widgets.Select(
            name='Select search method',
            options=self.options,
            value=self.method
        )
        self._widget_selector.param.watch(self._callback_selector, 'value', onlychanged=False)

        self._widget_search_active = pn.widgets.RadioButtonGroup(name='Start the search', options=['Start', 'Stop'], value='Stop')
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

        self._widget_activate_frame_capture = pn.widgets.Checkbox(name='Update frame realtime', value=self.activate_frame_capture)
        self._widget_activate_frame_capture.param.watch(self._callback_activate_frame_capture, 'value',
                                            onlychanged=False)

        self._widget_activate_gradient_descend = pn.widgets.Checkbox(name='Activate gradient descend',
                                                                  value=self.active_gradient_descend)
        self._widget_activate_gradient_descend.param.watch(self._callback_active_gradient_descend, 'value',
                                                        onlychanged=False)

        self._widget_refresh_frame = pn.widgets.Button(name="Manually refresh mesh", button_type="success")
        self._widget_refresh_frame.param.watch(self._callback_refresh_frame, 'clicks',
                                          onlychanged=False)

        self._widget_sleep = pn.widgets.FloatSlider(name='Step delay (seconds)',
                                                  value=self.sleep,
                                                  start=0.0,
                                                  end=5.0,
                                                    step=0.05)
        self._widget_sleep.param.watch(self._callback_sleep, 'value', onlychanged=False)

        self._widget_histogram_refresh = pn.widgets.Button(name="Manually refresh histogram", button_type="success")
        self._widget_histogram_refresh.param.watch(self._callbak_histogram_refresh, 'clicks',
                                               onlychanged=False)

        self._widget_variance = pn.widgets.Spinner(name='Proposed Variance', value=self.step_variance, step =10)
        self._widget_variance.param.watch(self._callback_variance, 'value', onlychanged=False)

        self._widget_speed_gradient_descend = pn.widgets.FloatSlider(name='Speed convergence',
                                                                     start=0.0,
                                                                     end=10.0,
                                                                     value=self.speed_alpha)
        self._widget_speed_gradient_descend.param.watch(self._callback_speed_gradient_descend, 'value', onlychanged=False)

        self._widget_number_samples = pn.widgets.Spinner(name='Number of samples', value=self.number_samples)
        self._widget_number_samples.param.watch(self._callback_number_samples, 'value',
                                                        onlychanged=False)

        self._widget_memory_steps = pn.widgets.Spinner(name='Memory steps', value=self.memory_steps)
        self._widget_memory_steps.param.watch(self._callback_memory_steps, 'value',
                                                onlychanged=False)

        self._widget_leapfrog_step = pn.widgets.FloatSlider(name='Leapfrog_step',
                                                            start=0.0,
                                                            end=10.0,
                                                            value=self.leapfrog_step)
        self._widget_leapfrog_step.param.watch(self._callback_leapfrog_step, 'value',
                                              onlychanged=False)

        self._widget_leapfrog_points = pn.widgets.Spinner(name='Leapfrog points', value=self.leapfrog_points)
        self._widget_leapfrog_points.param.watch(self._callback_leapfrog_points, 'value',
                                              onlychanged=False)

        self._widget_optimum = pn.widgets.Select(
            name='Select convergence to local optimum',
            options=["Maximum", "Minimum"],
            value=self.direction_search
        )
        self._widget_optimum.param.watch(self._callback_optimum, 'value', onlychanged=False)

        self._widget_margins_crop = pn.widgets.Spinner(name='Crop margins to reduce search area', value=self.margins_crop)
        self._widget_margins_crop.param.watch(self._callback_margins_crop, 'value',
                                                 onlychanged=False)


    def _callback_selector(self, event):
        self.x = None
        self.y = None
        self.x_list = []
        self.y_list = []
        self.method = event.new

    def _callback_search_active(self, event):
        if event.new == 'Start':
            self.search_active = True
        else:
            self.search_active = False

    def _callback_plot_points(self, event): self.plot_xy = event.new

    def _callback_plot_contour(self, event): self.plot_contour_xy = event.new

    def _callback_show_frame(self, event): self.show_frame = event.new

    def _callback_activate_frame_capture(self, event): self.activate_frame_capture = event.new

    def _callback_active_gradient_descend(self, event): self.active_gradient_descend = event.new

    def _callback_refresh_frame(self, event):
        self.lock.acquire()
        #self.pause()
        #frame = self.sensor.get_frame()
        #if self.crop:
        #    frame = self.crop_frame(frame)
        #    frame = self.clip_frame(frame)
        self.update_mesh(self.frame, margins_crop=self.margins_crop, fill_value=0)
        self.lock.release()
        #self.run()

    def _callback_sleep(self, event): self.sleep = event.new

    def _callbak_histogram_refresh(self, event):
        self.lock.acquire()
        self.histogram.object = self.hist_distribution()
        self.lock.acquire()

    def _callback_variance(self, event): self.step_variance = event.new

    def _callback_speed_gradient_descend(self, event): self.speed_alpha = event.new

    def _callback_number_samples(self, event): self.number_samples = event.new

    def _callback_memory_steps(self, event): self.memory_steps = event.new

    def _callback_leapfrog_step(self, event): self.leapfrog_step = event.new

    def _callback_leapfrog_points(self, event): self.leapfrog_points = event.new

    def _callback_optimum(self, event): self.direction_search = event.new

    def _callback_margins_crop(self, event): self.margins_crop = event.new