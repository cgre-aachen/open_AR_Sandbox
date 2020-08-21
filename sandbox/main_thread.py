from warnings import warn
import collections
import numpy
import threading
import panel as pn
pn.extension()
import matplotlib.pyplot as plt
import pandas as pd
import traceback

from sandbox.projector import Projector, ContourLinesModule, CmapModule
from sandbox.sensor import Sensor
from sandbox.markers import MarkerDetection


class MainThread:
    """
    Module with threading methods
    """
    def __init__(self, sensor: Sensor, projector: Projector, aruco: MarkerDetection = None, modules: list = [],
                 crop: bool = True, clip: bool = True, check_change: bool = False, **kwargs):
        """

        Args:
            sensor:
            projector:
            aruco:
            modules:
            crop:
            clip:
            check_change:
            **kwargs:
        """
        self.sensor = sensor
        self.projector = projector
        self.projector.ax.cla()
        self.extent = sensor.extent
        self.box = sensor.physical_dimensions
        self.contours = ContourLinesModule(extent=self.extent)
        self.cmap_frame = CmapModule(extent=self.extent)

        #start the modules
        self.modules = collections.OrderedDict({'cmap_frame': self.cmap_frame, 'contours': self.contours})

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
                          'marker': pd.DataFrame(),
                          'cmap': plt.cm.get_cmap('gist_earth'),
                          'norm': None,
                          'active_cmap': True,
                          'active_contours': True,
                          'same_frame': False,
                          'trigger': self.projector.trigger} #TODO: Carefull with this use because it can make to paint the figure incompletely

                          #'freeze_frame': False}

        self.previous_frame = self.sb_params['frame']
        #self.freeze_frame = False
        self.check_change = check_change
        self._rtol = 0.2
        self._atol = 5
        # render the frame
        self.cmap_frame.render_frame(self.sb_params['frame'], self.sb_params['ax'])
        # plot the contour lines
        self.contours.plot_contour_lines(self.sb_params['frame'], self.sb_params['ax'])
        self.projector.trigger()

    #@property @TODO: test if this works
    #def sb_params(self):
    #    return {'frame': self.sensor.get_frame(),
    #              'ax': self.projector.ax,
    #              'extent': self.sensor.extent,
    #              'marker': pd.DataFrame(),
    #              'cmap': plt.cm.get_cmap('gist_earth'),
    #              'norm': None,
    #              'active_cmap': True,
    #              'active_contours': True,
    #              'same_frame': False,
    #              'freeze_frame': False}

    def update(self, **kwargs):
        """
        Args:
            **kwargs:

        Returns:

        """
        #self.sb_params['ax'].cla()
        #self.delete_axes(self.sb_params['ax'])
        #self.sb_params['freeze_frame'] = self.freeze_frame
        #if self.sb_params['freeze_frame']:
        #    frame = self.previous_frame
        #else:
        frame = self.sensor.get_frame()
        self.sb_params['extent'] = self.sensor.extent
        #self.sb_params['ax'].set_xlim(xmin=self.sensor.extent[0], xmax=self.sensor.extent[1])
        #self.sb_params['ax'].set_ylim(ymin=self.sensor.extent[2], ymax=self.sensor.extent[3])
        #self.sb_params['cmap'] = self.cmap_frame.cmap
        # This is to avoid noise in the data
        if self.check_change:
            if not numpy.allclose(self.previous_frame, frame, atol=self._atol, rtol=self._rtol, equal_nan=True):
                self.previous_frame = frame
                self.sb_params['same_frame'] = False
            else:
                frame = self.previous_frame
                self.sb_params['same_frame'] = True
        else: self.sb_params['same_frame'] = False
        self.sb_params['frame'] = frame

        #filter
        if self.ARUCO_ACTIVE:
            df = self.Aruco.update()
        else:
            df = pd.DataFrame()
            plt.pause(0.1)

        self.sb_params['marker'] = df

        #TODO: Use the modules in a big try and except?
        try:
            self.lock.acquire()
            for key in self.modules.keys():
                self.modules[key].lock = self.lock
                self.sb_params = self.modules[key].update(self.sb_params)
            self.lock.release()
        except Exception:
            traceback.print_exc()
            self.lock.release()
            self.thread_status = 'stopped'

        self.sb_params['ax'].set_xlim(xmin=self.sb_params.get('extent')[0], xmax=self.sb_params.get('extent')[1])
        self.sb_params['ax'].set_ylim(ymin=self.sb_params.get('extent')[2], ymax=self.sb_params.get('extent')[3])
        #self.cmap_frame.update(self.sb_params)
        #plot the contour lines
        #self.contours.update(self.sb_params)
        if isinstance(self.Aruco, MarkerDetection):
            _ = self.Aruco.plot_aruco(self.sb_params['ax'], self.sb_params['marker'])
        self.lock.acquire()
        self.projector.trigger()
        self.lock.release()

    def add_module(self, name: str, module):
        """Add an specific module to run the update in the main thread"""
        self.modules[name] = module
        self.modules.move_to_end(name, last=False)
        print('module ' + name + ' added to modules')

    def remove_module(self, name: str):
        """Remove a current module from the main thread"""
        if name in self.modules.keys():
            self.modules.pop(name)
            print( 'module ' + name + ' removed')
        else:
            print('No module with name ' + name + ' was found')

    def delete_axes(self, ax):
        """
        #TODO: Need to find a better a way to delete the axes rather than ax.cla()
        Args:
            ax:
        Returns:

        """
        #self.cmap_frame.delete_image()
        #ax.cla()
        #self.extent = self.sensor.extent
        #ax.collections = []
        #ax.artists = []
        #ax.text = []
        pass

    def thread_loop(self):
        while self.thread_status == 'running':
            self.update()


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

    def widget_plot_module(self):
        if isinstance(self.Aruco, MarkerDetection):
            marker = pn.Column(self.widgets_aruco_visualization(), self.widget_thread_controller())
            widgets = pn.Column(self.cmap_frame.widgets_plot(),
                                self.contours.widgets_plot())
            rows = pn.Row(widgets, marker)
        else:
            widgets = pn.Column(self.cmap_frame.widgets_plot(),
                                self.contours.widgets_plot())
            rows = pn.Row(widgets, self.widget_thread_controller())

        panel = pn.Column("## Plotting interaction widgets", rows)
        return panel


    def widget_thread_controller(self):
        self._widget_thread_selector = pn.widgets.RadioButtonGroup(name='Thread controller',
                                                                   options=["Start", "Stop"],
                                                                   value="Start",
                                                                   button_type='success')
        self._widget_thread_selector.param.watch(self._callback_thread_selector, 'value', onlychanged=False)

        self._widget_check_difference = pn.widgets.Checkbox(name='Check changes in fame', value=self.check_change)
        self._widget_check_difference.param.watch(self._callback_check_difference, 'value',
                                                  onlychanged=False)

        #self._widget_freeze_frame = pn.widgets.Checkbox(name='Freeze frame acquisition', value=self.freeze_frame)
        #self._widget_freeze_frame.param.watch(self._callback_freeze_frame, 'value',
        #                                          onlychanged=False)

        panel = pn.Column("##<b>Thread Controller</b>",
                          self._widget_thread_selector,
                          self._widget_check_difference,
                          #self._widget_freeze_frame)
                          )
        return panel

    def _callback_check_difference(self, event):
        self.check_change = event.new

    #def _callback_freeze_frame(self, event):
    #    self.freeze_frame = event.new

    def _callback_thread_selector(self, event):
        if event.new == "Start":
            self.run()
        elif event.new == "Stop":
            self.stop()

    def widgets_aruco_visualization(self):
        self._widget_aruco = pn.widgets.Checkbox(name='Aruco Detection', value=self.ARUCO_ACTIVE)
        self._widget_aruco.param.watch(self._callback_aruco, 'value',
                                       onlychanged=False)
        panel = pn.Column("## Activate aruco detetection", self._widget_aruco, self.Aruco.widgets_aruco())
        return panel

    def _callback_aruco(self, event):
        self.ARUCO_ACTIVE = event.new