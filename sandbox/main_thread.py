from warnings import warn
import numpy
import threading
import panel as pn
import matplotlib.pyplot as plt

from sandbox.projector import Projector, ContourLinesModule, CmapModule
from sandbox.sensor import Sensor
from sandbox.markers import ArucoMarkers
from sandbox.modules import *

class MainThread:
    """
    Module with threading methods
    """
    def __init__(self, sensor: Sensor, projector: Projector, aruco: ArucoMarkers = None, modules: list = [],
                 crop: bool = True, clip: bool = True, **kwargs):

        self.sensor = sensor
        self.projector = projector
        self.contours = ContourLinesModule(extent=sensor.extent)
        self.cmap_frame = CmapModule(extent=sensor.extent)

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
        if isinstance(self.Aruco, ArucoMarkers):
            self.ARUCO_ACTIVE = True

        ax = self.projector.ax
        frame = self.sensor.get_frame()
        # render the frame
        self.cmap_frame.render_frame(frame, ax, **kwargs)
        # plot the contour lines
        self.contours.plot_contour_lines(frame, ax, **kwargs)
        self.projector.trigger()

    def update(self, **kwargs):
        ax = self.projector.ax
        self.delete_axes(ax):



        modules = self.modules
        frame = self.sensor.get_frame()
        #filter
        if self.ARUCO_ACTIVE:
            points = self.aruco.get_loaction()
        else: points = []

        for m in modules:
            frame, ax, cmap = m.update(frame, ax, points, **kwargs)

        self.cmap_frame.update(frame, ax)
        #plot the contour lines
        self.contours.plot_contour_lines(frame, ax, **kwargs)

        self.projector.trigger()

    def delete_axes(self, ax):
        self.cmap_frame.delete_image()


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