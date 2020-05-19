# TODO: Load all the modules from here!!! Important for testing (?)

import logging
import panel as pn
import traceback

from sandbox.calibration.calibration import CalibrationData
from sandbox.sensor.sensor_api import KinectV2
from sandbox.projector.projector import Projector

from sandbox.markers.aruco import ArucoMarkers

from sandbox.calibration.calibration_module import CalibModule
from sandbox.modules import GradientModule, LandslideSimulation, LoadSaveTopoModule, TopoModule,\
    PrototypingModule
from sandbox.modules.gempy.gempy_module import GemPyModule

# logging and exception handling
verbose = False
if verbose:
    logging.basicConfig(filename="main.log",
                        filemode='w',
                        level=logging.WARNING,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        )

def calibrate_sandbox(aruco_marker=True):
    calib = CalibrationData(p_width=1280, p_height=800)
    sensor = KinectV2(calib)
    projector = Projector(calib)
    if aruco_marker:
        aruco = ArucoMarkers(sensor, calib)
        module = CalibModule(calib, sensor, projector, Aruco=aruco)
    else:
        module = CalibModule(calib, sensor, projector)
    module.setup()
    module.run()
    module.calibrate().show()

def start_server(calibration_file, aruco_marker=True, geo_model=None):

    calib = CalibrationData(file=calibration_file)
    sensor = KinectV2(calib)
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

    def __init__(self, calibrationdata, sensor, projector, Aruco=None, geo_model=None, **kwargs):
        self.calib = calibrationdata
        self.sensor = sensor
        self.projector = projector
        self.aruco = Aruco
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
        elif module == 'GempyModule':
            try:
                if self.geo_model is not None:
                    self.module = GemPyModule(self.geo_model, self.calib, self.sensor, self.projector, self.aruco)
                else:
                    print("No geo_model found. Please pass a valid gempy model"
                          "")
            except Exception:
                traceback.print_exc()
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

        self._widget_topo = pn.widgets.Button(name="LoadSaveTopoModule", button_type="success")
        self._widget_topo.param.watch(self._callback_topo, 'clicks',
                                          onlychanged=False)

        self._widget_landslide = pn.widgets.Button(name="LandslideSimulation", button_type="success")
        self._widget_landslide.param.watch(self._callback_landslide, 'clicks',
                                      onlychanged=False)

        self._widget_gempy = pn.widgets.Button(name="GempyModule", button_type="success")
        self._widget_gempy.param.watch(self._callback_gempy, 'clicks',
                                           onlychanged=False)

        return True

    def show_widgets(self):
        self._create_widgets()
        widgets = pn.WidgetBox('<b>Select the module you want to start </b>',
                               self._widget_topo,
                               self._widget_gradient,
                               self._widget_load_save_topo,
                               self._widget_landslide,
                               self._widget_gempy)
        panel = pn.Column("### Module selector", widgets)

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
        print("didi it worked?")
        self.module.run()
















