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
from sandbox.sensor.kinectV2 import _platform
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
        #TODO: in all the modules be carefull with zorder
        self.sensor = sensor
        self.projector = projector
        self.projector.clear_axes()
        self.contours = ContourLinesModule(extent=self.sensor.extent)
        self.cmap_frame = CmapModule(extent=self.sensor.extent)

        #start the modules
        #self.modules = collections.OrderedDict({'CmapModule': self.cmap_frame, 'ContourLinesModule': self.contours})
        self.modules = {'CmapModule': self.cmap_frame, 'ContourLinesModule': self.contours}
        self._modules = {'CmapModule': self.cmap_frame, 'ContourLinesModule': self.contours} #cachee

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
                          'box_dimensions': self.sensor.physical_dimensions,
                          'marker': pd.DataFrame(),
                          'cmap': plt.cm.get_cmap('gist_earth'),
                          'norm': None,
                          'active_cmap': True,
                          'active_contours': True,
                          'same_frame': False,
                          'lock_thread': self.lock,
                          'trigger': self.projector.trigger} #TODO: Carefull with this use because it can make to paint the figure incompletely

                          #'freeze_frame': False}

        self.previous_frame = self.sb_params['frame']

        #To reduce the noise of the data
        self.check_change = check_change
        self._rtol = 0.2
        self._atol = 5
        # render the frame
        self.cmap_frame.render_frame(self.sb_params['frame'], self.sb_params['ax'])
        # plot the contour lines
        self.contours.plot_contour_lines(self.sb_params['frame'], self.sb_params['ax'])
        self.projector.trigger()

        self._create_widgets()

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
        self.sb_params['ax'] = self.projector.ax

        frame = self.sensor.get_frame()
        self.sb_params['extent'] = self.sensor.extent

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
        self.lock.acquire()
        if self.ARUCO_ACTIVE:
            df = self.Aruco.update()
        else:
            df = pd.DataFrame()
            plt.pause(0.2)
        self.lock.release()

        self.sb_params['marker'] = df

        #TODO: Use the modules in a big try and except?
        try:
            self.lock.acquire()
            for key in list(self.modules.keys()): #reversed so the contourlines and cmap that were added first, get painted at last
                self.sb_params = self.modules[key].update(self.sb_params)
            self.lock.release()
        except Exception:
            traceback.print_exc()
            self.lock.release()
            self.thread_status = 'stopped'

        self.sb_params['ax'].set_xlim(xmin=self.sb_params.get('extent')[0], xmax=self.sb_params.get('extent')[1])
        self.sb_params['ax'].set_ylim(ymin=self.sb_params.get('extent')[2], ymax=self.sb_params.get('extent')[3])

        if isinstance(self.Aruco, MarkerDetection):
            _ = self.Aruco.plot_aruco(self.sb_params['ax'], self.sb_params['marker'])

        self.lock.acquire()
        self.projector.trigger()
        self.lock.release()

    def add_module(self, name: str, module):
        """Add an specific module to run the update in the main thread"""
        self.modules[name] = module
        self._modules[name] = module
        #self.modules.move_to_end(name, last=True)
        print('module ' + name + ' added to modules')

    def remove_module(self, name: str):
        """Remove a current module from the main thread"""
        if name in self.modules.keys():
            self.lock.acquire()
            self.modules.pop(name)
            print( 'module ' + name + ' removed')
            self.lock.release()
        else:
            print('No module with name ' + name + ' was found')

    def module_manager(self, active_modules: list = []):
        #add a module
        for name_module in active_modules:
            if name_module not in self.modules:
                self.add_module(name=name_module, module=self._modules[name_module])
        #delete the module
        if len(active_modules) > 0:
            [self.remove_module(name) for name in self.modules if name not in active_modules]

        if self._widget_module_selector is not None:
            self._widget_module_selector.options = list(self._modules.keys())
            self._widget_module_selector.value = list(self.modules.keys())

    def thread_loop(self):
        while self.thread_status == 'running':
            self.update()


    def run(self):
        if self.thread_status != 'running':
            if _platform == "Linux" and self.sensor.s_name == "kinect_v2":
                self.sensor.Sensor._run()
            self.thread_status = 'running'
            self.thread = threading.Thread(target=self.thread_loop, daemon=True, )
            self.thread.start()
            print('Thread started or resumed...')

        else:
            print('Thread already running.')

    def stop(self):
        if self.thread_status is not 'stopped':
            if _platform == "Linux" and self.sensor.s_name == "kinect_v2":
                self.sensor.Sensor._stop()
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
            widgets = pn.Column(self.cmap_frame.show_widgets(),
                                self.contours.show_widgets())
            rows = pn.Row(widgets, marker)
        else:
            widgets = pn.Column(self.cmap_frame.show_widgets(),
                                self.contours.show_widgets())
            rows = pn.Row(widgets, self.widget_thread_controller())

        panel = pn.Column("## Plotting interaction widgets", rows)
        return panel


    def widget_thread_controller(self):
        panel = pn.Column("##<b>Thread Controller</b>",
                          self._widget_thread_selector,
                          self._widget_check_difference,
                          self._widget_module_selector,
                          self._widget_clear_axes
                          #self._widget_freeze_frame)
                          )
        return panel

    def _create_widgets(self):
        self._widget_thread_selector = pn.widgets.RadioButtonGroup(name='Thread controller',
                                                                   options=["Start", "Stop"],
                                                                   value="Start",
                                                                   button_type='success')
        self._widget_thread_selector.param.watch(self._callback_thread_selector, 'value', onlychanged=False)

        self._widget_check_difference = pn.widgets.Checkbox(name='Check changes in fame', value=self.check_change)
        self._widget_check_difference.param.watch(self._callback_check_difference, 'value',
                                                  onlychanged=False)

        self._widget_clear_axes = pn.widgets.Button(name="Clear axes from projector / refresh list", button_type="warning")
        self._widget_clear_axes.param.watch(self._callback_clear_axes, 'clicks',
                                            onlychanged=False)

        self._widget_module_selector = pn.widgets.CrossSelector(name="Module manager",
                                                                value=list(self.modules.keys()),
                                                                options=list(self._modules.keys()),
                                                                definition_order=False)
        self._widget_module_selector.param.watch(self._callback_module_selector, 'value',
                                                 onlychanged=False)


    def _callback_clear_axes(self, event):
        self.projector.clear_axes()
        self._widget_module_selector.options = list(self._modules.keys())
        self._widget_module_selector.value = list(self.modules.keys())

    def _callback_check_difference(self, event):
        self.check_change = event.new

    def _callback_module_selector(self, event):
        self.module_manager(event.new)

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