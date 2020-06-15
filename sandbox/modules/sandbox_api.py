# TODO: Load all the modules from here!!! Important for testing (?)

import logging
import panel as pn
import traceback
import json

from sandbox.calibration.calibration import CalibrationData
from sandbox.sensor.sensor_api import Sensor
from sandbox.projector.projector import Projector

from sandbox.markers.aruco import ArucoMarkers

from sandbox.calibration.calibration_module import CalibModule
from sandbox.modules import GradientModule, LandslideSimulation, LoadSaveTopoModule, TopoModule, SearchMethodsModule
from sandbox.modules.gempy.gempy_module import GemPyModule

# logging and exception handling
verbose = False
if verbose:
    logging.basicConfig(filename="main.log",
                        filemode='w',
                        level=logging.WARNING,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        )

def calibrate_sandbox(sensor_name='kinect_v2', aruco_marker=True):
    calib = CalibrationData(p_width=1280, p_height=800)
    sensor = Sensor(calib, sensor_name)
    projector = Projector(calib)
    if aruco_marker:
        aruco = ArucoMarkers(sensor, calib)
        module = CalibModule(calib, sensor, projector, Aruco=aruco)
    else:
        module = CalibModule(calib, sensor, projector)
    module.setup()
    module.run()
    module.show_widgets().show()

def start_server(calibration_file=None, sensor_name='kinect_v2', aruco_marker=True, geo_model=None):
    if calibrate_sandbox is None:
        calib = CalibrationData(p_width=1280, p_height=800)
    else:
        calib = CalibrationData(file=calibration_file)

    sensor = Sensor(calib, sensor_name)
    projector = Projector(calib)
    if aruco_marker:
        aruco = ArucoMarkers(sensor, calib)
        if geo_model is not None:
            module = Sandbox(calib, sensor, projector, aruco=aruco, geo_model=geo_model)
        else:
            module = Sandbox(calib, sensor, projector, aruco=aruco)
    else:
        if geo_model is not None:
            module = Sandbox(calib, sensor, projector, geo_model=geo_model)
        else:
            module = Sandbox(calib, sensor, projector)

    module.start()

    return True


class Sandbox:
    # Wrapping API-class

    def __init__(self, calibrationdata, sensor, projector, aruco=None, geo_model=None, **kwargs):
        self.calib = calibrationdata
        self.sensor = sensor
        self.projector = projector
        self.aruco = aruco
        if isinstance(self.aruco, ArucoMarkers):
            self.disable_aruco = False
            self.ARUCO_ACTIVE = True
        else:
            self.disable_aruco = True
            self.ARUCO_ACTIVE = False

        self.geo_model = geo_model

        self.module = None
        self.module_active = False

    def setup_server(self, module):
        if module == 'GradientModule':
            self.module = GradientModule(self.calib, self.sensor, self.projector, self.aruco)
        elif module == 'LandslideSimulation':
            self.module = LandslideSimulation(self.calib, self.sensor, self.projector, self.aruco)
        elif module == 'LoadSaveTopoModule':
            self.module = LoadSaveTopoModule(self.calib, self.sensor, self.projector, self.aruco)
        elif module == 'TopoModule':
            self.module = TopoModule(self.calib, self.sensor, self.projector, self.aruco)
        elif module == 'SearchMethodsModule':
            self.module = SearchMethodsModule(self.calib, self.sensor, self.projector, self.aruco)
        elif module == 'GempyModule':
            try:
                if self.geo_model is not None:
                    self.module = GemPyModule(self.geo_model, self.calib, self.sensor, self.projector, self.aruco)
                else:
                    print("No geo_model found. Please pass a valid gempy model")
            except Exception:
                traceback.print_exc()
        elif module == 'CalibModule':
            self.module = CalibModule(self.calib, self.sensor, self.projector, self.aruco)

        self.module_active = True
        self.module.setup()
        self.module.update()
        self.module.run()

        try:
            self.module.show_widgets().show()
        except Exception:
            traceback.print_exc()

    def start(self):
        self.show_widgets().show()

    def _create_widgets(self):
        self._widget_gradient = pn.widgets.Button(name="GradientModule", button_type="success")
        self._widget_gradient.param.watch(self._callback_gradient, 'clicks',
                                          onlychanged=False)

        self._widget_load_save_topo = pn.widgets.Button(name="LoadSaveTopoModule", button_type="success")
        self._widget_load_save_topo.param.watch(self._callback_load_save, 'clicks',
                                          onlychanged=False)

        self._widget_topo = pn.widgets.Button(name="TopoModule", button_type="success")
        self._widget_topo.param.watch(self._callback_topo, 'clicks',
                                          onlychanged=False)

        self._widget_landslide = pn.widgets.Button(name="LandslideSimulation", button_type="success")
        self._widget_landslide.param.watch(self._callback_landslide, 'clicks',
                                      onlychanged=False)

        self._widget_search = pn.widgets.Button(name="SearchMethodsModule", button_type="success")
        self._widget_search.param.watch(self._callback_search, 'clicks',
                                      onlychanged=False)

        self._widget_gempy = pn.widgets.Button(name="GempyModule", button_type="success")
        self._widget_gempy.param.watch(self._callback_gempy, 'clicks',
                                           onlychanged=False)

        self._widget_calibration = pn.widgets.FileInput(name="Load calibration", accept=".json")
        self._widget_calibration.param.watch(self._callback_calibration, 'value')

        self._widget_create_calibration = pn.widgets.Button(name="CalibModule", button_type="success")
        self._widget_create_calibration.param.watch(self._callback_create_calibration, 'clicks',
                                       onlychanged=False)

        self._widget_new_server = pn.widgets.Button(name="New Server", button_type="warning")
        self._widget_new_server.param.watch(self._callback_new_server, 'clicks',
                                       onlychanged=False)

        self._widget_thread_selector = pn.widgets.RadioButtonGroup(name='Thread controller',
                                                                   options=["Start", "Stop"],
                                                                   value="Start",
                                                                   button_type='success')
        self._widget_thread_selector.param.watch(self._callback_thread_selector, 'value', onlychanged=False)

        self._widget_aruco = pn.widgets.Checkbox(name='Aruco Detection', value=self.ARUCO_ACTIVE, disabled=self.disable_aruco)
        self._widget_aruco.param.watch(self._callback_aruco, 'value',
                                       onlychanged=False)

        return True

    def show_widgets(self):
        self._create_widgets()
        widgets = pn.Column("# Module selector",
                            '<b>Select the module you want to start </b>',
                            self._widget_topo,
                            self._widget_gradient,
                            self._widget_load_save_topo,
                            self._widget_landslide,
                            self._widget_search,
                            self._widget_gempy,
                            '<b>Change the calibration file</b>',
                            self._widget_calibration,
                            '<b>Create new calibration file</b>',
                            self._widget_create_calibration)
        thread = pn.Column("##<b>Thread Controller</b>",
                           self._widget_thread_selector,
                           '<b>Start a new server</b>',
                           self._widget_new_server,
                           "<b>Deactivate or activate aruco detection</b>",
                           self._widget_aruco)

        panel = pn.Row(widgets, thread)

        return panel

    def _callback_gradient(self, event):
        if self.module_active:
            self.module.stop()
        self.setup_server('GradientModule')

    def _callback_load_save(self, event):
        if self.module_active:
            self.module.stop()
        self.setup_server('LoadSaveTopoModule')

    def _callback_topo(self, event):
        if self.module_active:
            self.module.stop()
        self.setup_server('TopoModule')

    def _callback_landslide(self, event):
        if self.module_active:
            self.module.stop()
        self.setup_server('LandslideSimulation')

    def _callback_gempy(self, event):
        if self.module_active:
            self.module.stop()
        self.setup_server('GempyModule')
        self.module.run()

    def _callback_calibration(self, event):
        data_bytes = event.new
        data_decode = data_bytes.decode()
        data_dict = json.loads(data_decode)
        self.calib.__dict__ = data_dict
        self.sensor.__init__(self.calib)
        self.projector.__init__(self.calib)

    def _callback_create_calibration(self, event):
        if self.module_active:
            self.module.stop()
        self.setup_server('CalibModule')

    def _callback_new_server(self, event):
        self.projector.__init__(self.calib)

    def _callback_thread_selector(self, event):
        if event.new == "Start":
            self.module.run()
        elif event.new == "Stop":
            self.module.stop()

    def _callback_aruco(self, event):
        self.module.ARUCO_ACTIVE = event.new

    def _callback_search(self, event):
        if self.module_active:
            self.module.stop()
        self.setup_server('SearchMethodsModule')










