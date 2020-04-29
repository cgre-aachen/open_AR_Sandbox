from abc import ABCMeta
from abc import abstractmethod
import numpy
import threading
from sandbox.plot.plot import Plot

class Module(object):
    """
    Parent Module with threading methods and abstract attributes and methods for child classes
    """
    __metaclass__ = ABCMeta

    def __init__(self, calibrationdata, sensor, projector, Aruco=None, crop=True, clip = True, norm = False, **kwargs):
        self.calib = calibrationdata
        self.sensor = sensor
        self.projector = projector
        self.plot = Plot(self.calib, **kwargs)

        # flags
        self.crop = crop
        self.clip = clip
        self.norm = norm # for TopoModule to scale the topography

        # threading
        self.lock = threading.Lock()
        self.thread = None
        self.thread_status = 'stopped'  # status: 'stopped', 'running', 'paused'

        # connect to ArucoMarker class
        # if CV2_IMPORT is True:
        self.Aruco = Aruco

    @abstractmethod
    def setup(self):
        # Wildcard: Everything necessary to set up before a model update can be performed.
        pass

    @abstractmethod
    def update(self):
        # Wildcard: Single model update operation that can be looped in a thread.
        pass

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

    def depth_mask(self, frame):
        """ Creates a boolean mask with True for all values within the set sensor range and False for every pixel
        above and below. If you also want to use clipping, make sure to use the mask before.
        """

        mask = numpy.ma.getmask(numpy.ma.masked_outside(frame, self.calib.s_min, self.calib.s_max))
        return mask

    def crop_frame(self, frame):
        """ Crops the data frame according to the horizontal margins set up in the calibration
        """

        # TODO: Does not work yet for s_top = 0 and s_right = 0, which currently returns an empty frame!
        # TODO: Workaround: do not allow zeroes in calibration widget and use default value = 1
        # TODO: File numpy issue?
        crop = frame[self.calib.s_bottom:-self.calib.s_top, self.calib.s_left:-self.calib.s_right]
        return crop

    def crop_frame_workaround(self, frame):
        # bullet proof working example
        if self.calib.s_top == 0 and self.calib.s_right == 0:
            crop = frame[self.calib.s_bottom:, self.calib.s_left:]
        elif self.calib.s_top == 0:
            crop = frame[self.calib.s_bottom:, self.calib.s_left:-self.calib.s_right]
        elif self.calib.s_right == 0:
            crop = frame[self.calib.s_bottom:-self.calib.s_top, self.calib.s_left:]
        else:
            crop = frame[self.calib.s_bottom:-self.calib.s_top, self.calib.s_left:-self.calib.s_right]

        return crop

    def clip_frame(self, frame):
        """ Clips all values outside of the sensor range to the set s_min and s_max values.
        If you want to create a mask make sure to call depth_mask before performing the clip.
        """

        clip = numpy.clip(frame, self.calib.s_min, self.calib.s_max)
        return clip
