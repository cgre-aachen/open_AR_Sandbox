from warnings import warn
import numpy
import threading
import panel as pn
pn.extension()
import matplotlib.pyplot as plt
import pandas as pd

from sandbox.projector import Projector, ContourLinesModule, CmapModule
from sandbox.sensor import Sensor
from sandbox.markers import MarkerDetection
from sandbox.modules import *

class MainThread:
    """
    Module with threading methods
    """
    def __init__(self, sensor: Sensor, projector: Projector, aruco: MarkerDetection = None, modules: list = [],
                 crop: bool = True, clip: bool = True, **kwargs):

        self.check_change = True
        self.sensor = sensor
        self.projector = projector
        self.extent = sensor.extent
        self.box = sensor.physical_dimensions
        self.contours = ContourLinesModule(extent=self.extent)
        self.cmap_frame = CmapModule(extent=self.extent)

        #start the modules
        self.modules = modules

        # flags
        self.crop = crop
        self.clip = clip

        # threading
        self.lock = threading.Lock()
        self.thread = None
        self.thread_status = 'stopped'  # status: 'stopped', 'running', 'paused'

        # connect to ArucoMarker class
        # if CV2_IMPORT is True:
        self.Aruco = aruco
        self.ARUCO_ACTIVE = False
        if isinstance(self.Aruco, MarkerDetection):
            self.ARUCO_ACTIVE = True

        self.sb_params = {'frame': self.sensor.get_frame(),
                          'ax': self.projector.ax,
                          'extent': self.sensor.extent,
                          'marker': [],
                          'cmap': plt.cm.get_cmap('gist_earth'),
                          'norm': None,
                          'active_cmap': True,
                          'active_contours': True}

        self.previous_frame = self.sb_params['frame']
        # render the frame
        self.cmap_frame.render_frame(self.sb_params['frame'], self.sb_params['ax'])
        # plot the contour lines
        self.contours.plot_contour_lines(self.sb_params['frame'], self.sb_params['ax'])
        self.projector.trigger()

    def update(self, **kwargs):
        """
        Args:
            **kwargs:

        Returns:

        """
        self.sb_params['ax'].cla()
        #self.delete_axes(ax)
        frame = self.sensor.get_frame()
        self.sb_params['extent'] = self.sensor.extent

        # This is to avoid noise in the data
        if self.check_change:
            if not numpy.allclose(self.previous_frame, frame, atol=5, rtol=1e-1, equal_nan=True):
                self.previous_frame = frame
            else:
                frame = self.previous_frame
        self.sb_params['frame'] = frame

        #filter
        if self.ARUCO_ACTIVE:
            df = self.Aruco.update()
        else:
            df = pd.Dataframe()
        self.sb_params['marker'] = df
        modules = self.modules
        for m in modules:
            self.sb_params = m.update(self.sb_params)

        self.cmap_frame.update(self.sb_params)
        #plot the contour lines
        self.contours.update(self.sb_params)
        if self.ARUCO_ACTIVE:
            _ = self.Aruco.plot_aruco(self.sb_params['ax'], self.sb_params['df'])
        self.projector.trigger()

    def delete_axes(self, ax):
        """
        #TODO: Need to find a better a way to delete the axes rather than ax.cla()
        Args:
            ax:
        Returns:

        """
        #self.cmap_frame.delete_image()
        ax.cla()
        #self.extent = self.sensor.extent

    def thread_loop(self):
        while self.thread_status == 'running':
            self.lock.acquire()
            self.update()
            self.lock.release()

    def run(self):
        if self.thread_status != 'running':
            self.thread_status = 'running'
            self.thread = threading.Thread(target=self.thread_loop, daemon=True, )
            self.thread.start()
            print('Thread started or resumed...')
        else:
            print('Thread already running.')

    def stop(self):
        if self.thread_status is not 'stopped':
            self.thread_status = 'stopped'  # set flag to end thread loop
            self.thread.join()  # wait for the thread to finish
            print('Thread stopped.')
        else:
            print('thread was not running.')

    def pause(self):
        if self.thread_status == 'running':
            self.thread_status = 'paused'  # set flag to end thread loop
            self.thread.join()  # wait for the thread to finish
            print('Thread paused.')
        else:
            print('There is no thread running.')

    def resume(self):
        if self.thread_status != 'stopped':
            self.run()
        else:
            print('Thread already stopped.')