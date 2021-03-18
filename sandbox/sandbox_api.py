import panel as pn
import traceback
from sandbox import _calibration_dir
import platform
_platform = platform.system()

# Store the name of the sensor as a global variable, and the projector resolution
# so it can be used and changed in the same session for all the functions
name_sensor = "kinect_v2"
p_width = 1280
p_height = 800


def calibrate_projector():
    from sandbox.projector import Projector
    proj = Projector(use_panel=True)
    widget = proj.calibrate_projector()
    widget.show()
    return proj


def calibrate_sensor(calibprojector: str = _calibration_dir + "my_projector_calibration.json",
                     name: str = None):
    global name_sensor
    if name is None:
        name = name_sensor
    else:
        name_sensor = name
    from sandbox.sensor import CalibSensor
    module = CalibSensor(calibprojector=calibprojector, name=name)
    widget = module.calibrate_sensor()
    widget.show()
    return module.sensor


def start_server(calibprojector: str = None,  # _calibration_dir + "my_projector_calibration.json",
                 calibsensor: str = None,  # _calibration_dir + "my_sensor_calibration.json",
                 sensor_name: str = None,
                 aruco_marker: bool = True,
                 kwargs_external_modules: dict = {},
                 kwargs_gempy_module: dict = {},
                 kwargs_projector: dict = {},
                 kwargs_sensor: dict = {},
                 kwargs_aruco: dict = {},
                 ):
    global name_sensor, p_width, p_height
    if sensor_name is None:
        sensor_name = name_sensor
    else:
        name_sensor = sensor_name

    from sandbox.projector import Projector
    if kwargs_projector.get("p_width") is not None:
        p_width = kwargs_projector.get("p_width")
    if kwargs_projector.get("p_height") is not None:
        p_height = kwargs_projector.get("p_height")
    projector = Projector(calibprojector=calibprojector, use_panel=True, **kwargs_projector)

    from sandbox.sensor import Sensor
    sensor = Sensor(calibsensor=calibsensor, name=sensor_name, **kwargs_sensor)

    if aruco_marker:
        from sandbox.markers import MarkerDetection
        aruco = MarkerDetection(sensor=sensor, **kwargs_aruco)
    else:
        aruco = None

    module = Sandbox(sensor=sensor,
                     projector=projector,
                     aruco=aruco,
                     kwargs_gempy_module=kwargs_gempy_module,
                     kwargs_external_modules=kwargs_external_modules)

    module.start()

    return module


class Sandbox:
    """
    Wrapping API-class
    """

    def __init__(self,
                 sensor,  # : Sensor,
                 projector,  # : Projector,
                 aruco,  # : MarkerDetection = None,
                 kwargs_contourlines: dict = {},
                 kwargs_cmap: dict = {},
                 kwargs_external_modules: dict = {},
                 kwargs_gempy_module: dict = {},
                 ):
        self._gempy_import = False
        self._devito_import = False
        self._pygimli_import = False
        self._torch_import = False
        self._check_import(**kwargs_external_modules)

        self.sensor = sensor
        self._sensor_calib = self.sensor.json_filename
        self.projector = projector
        self._projector_calib = self.projector.json_filename
        self.aruco = aruco
        from sandbox.markers import MarkerDetection
        if isinstance(self.aruco, MarkerDetection):
            self._disable_aruco = False
            self.ARUCO_ACTIVE = True
        else:
            self._disable_aruco = True
            self.ARUCO_ACTIVE = False

        self.Modules = None
        self.module_active = False
        self.load_modules(kwargs_gempy_module=kwargs_gempy_module, **kwargs_external_modules)
        self.Main_Thread = self.start_main_thread(kwargs_contourlines=kwargs_contourlines,
                                                  kwargs_cmap=kwargs_cmap)
        self.lock = self.Main_Thread.lock

        return print('Sandbox server ready')

    def _check_import(self,
                      gempy_module: bool = False,
                      gimli_module: bool = False,
                      torch_module: bool = False,
                      devito_module: bool = False,
                      ):
        if gempy_module:
            try:  # Importing Gempy for GempyModule
                import gempy
                self._gempy_import = True
                del gempy
            except Exception as e:
                pass

        if _platform == "Linux":
            if devito_module:
                try:  # Importing Devito for SeismicModule - Only working for Linux system
                    import devito
                    self._devito_import = True
                    del devito
                except Exception as e:
                    pass
            else:
                _devito_import = False

        if gimli_module:
            try:  # Importing pygimli for GeoelectricsModule
                import pygimli
                self._pygimli_import = True
                del pygimli
            except Exception as e:
                pass
        if torch_module:
            try:  # Importing pytorch for LandscapeGeneration
                import torch
                self._torch_import = True
                del torch
            except Exception as e:
                pass

    def load_modules(self,
                     gempy_module: bool = False,
                     gimli_module: bool = False,
                     torch_module: bool = False,
                     devito_module: bool = False,
                     kwargs_gempy_module: dict = {},
                     ):
        from sandbox.modules import (TopoModule, GradientModule, LoadSaveTopoModule, LandslideSimulation,
                                     SearchMethodsModule)
        from sandbox.projector import ContourLinesModule, CmapModule
        self.Modules = {'ContourLinesModule': ContourLinesModule(extent=self.sensor.extent),
                        'CmapModule': CmapModule(extent=self.sensor.extent),
                        'TopoModule': TopoModule(extent=self.sensor.extent),
                        'GradientModule': GradientModule(extent=self.sensor.extent),
                        'LoadSaveTopoModule': LoadSaveTopoModule(extent=self.sensor.extent),
                        'LandslideSimulation': LandslideSimulation(extent=self.sensor.extent),
                        'SearchMethodsModule': SearchMethodsModule(extent=self.sensor.extent)}
        #self.Modules['SearchMethodsModule'].update_mesh(self.sensor.get_frame(),
        #                                                margins_crop=self.Modules['SearchMethodsModule'].margins_crop,
        #                                                fill_value=0)
        #self.Modules['SearchMethodsModule'].activate_frame_capture = False
        if gempy_module and self._gempy_import:
            from sandbox.modules.gempy import GemPyModule
            self.Modules['GemPyModule'] = GemPyModule(extent=self.sensor.extent,
                                                      box=self.sensor.physical_dimensions,
                                                      **kwargs_gempy_module)
        if devito_module and self._devito_import:
            from sandbox.modules.devito import SeismicModule
            self.Modules['SeismicModule'] = SeismicModule(extent=self.sensor.extent)

        if gimli_module and self._pygimli_import:
            from sandbox.modules.gimli import GeoelectricsModule
            self.Modules['GeoelectricsModule'] = GeoelectricsModule(extent=self.sensor.extent)

        if torch_module and self._torch_import:
            from sandbox.modules.pytorch import LandscapeGeneration
            self.Modules['LandscapeGeneration'] = LandscapeGeneration(extent=self.sensor.extent)

    def start_main_thread(self, kwargs_contourlines: dict = {}, kwargs_cmap: dict = {}):
        from sandbox.main_thread import MainThread
        thread = MainThread(sensor=self.sensor, projector=self.projector, aruco=self.aruco,
                            kwargs_contourlines=kwargs_contourlines, kwargs_cmap=kwargs_cmap)
        thread._modules = self.Modules
        thread.run()
        return thread

    def add_to_main_thread(self, module_name: str):
        if module_name in self.Modules:
            self.Main_Thread.add_module(module_name, self.Modules[module_name])
            try:
                new_tab = pn.Tabs((module_name, self.Modules[module_name].show_widgets()),
                                  ('Main Thread Controller', self.Main_Thread.widget_plot_module()))
                new_tab.show()
            except Exception:
                traceback.print_exc()
        else:
            print('No module to add with name: ', module_name)

    def remove_from_main_thread(self, module_name: str = None):
        if module_name is not None:
            if module_name in self.Modules:
                self.Main_Thread.remove_module(module_name)
            else:
                print('No module to remove with name: ', module_name)
        else:
            # TODO: Is this the best way to delete the modules except the colormap and the contourlines?
            #all = self.Main_Thread.modules.keys()
            for name in list(self.Main_Thread.modules.keys()):
                if name != 'CmapModule' and name != 'ContourLinesModule':
                    self.Main_Thread.remove_module(name)
            self.projector.clear_axes()

    def start(self):
        #self.Main_Thread.run()
        self.show_widgets().show()
        self.Main_Thread._update_widget_module_selector()

    def _create_widgets(self):
        # Local Modules
        self._widget_main_thread = pn.widgets.Button(name="MainThread_Controllers", button_type="success")
        self._widget_main_thread.param.watch(self._callback_main_thread, 'clicks', onlychanged=False)

        self._widget_gradient = pn.widgets.Button(name="GradientModule", button_type="success")
        self._widget_gradient.param.watch(self._callback_gradient, 'clicks', onlychanged=False)

        self._widget_load_save_topo = pn.widgets.Button(name="LoadSaveTopoModule", button_type="success")
        self._widget_load_save_topo.param.watch(self._callback_load_save, 'clicks', onlychanged=False)

        self._widget_topo = pn.widgets.Button(name="TopoModule", button_type="success")
        self._widget_topo.param.watch(self._callback_topo, 'clicks', onlychanged=False)

        self._widget_landslide = pn.widgets.Button(name="LandslideSimulation", button_type="success")
        self._widget_landslide.param.watch(self._callback_landslide, 'clicks', onlychanged=False)

        self._widget_search = pn.widgets.Button(name="SearchMethodsModule", button_type="success")
        self._widget_search.param.watch(self._callback_search, 'clicks', onlychanged=False)

        # External Modules
        self._widget_gempy = pn.widgets.Button(name="GempyModule", button_type="success")
        self._widget_gempy.param.watch(self._callback_gempy, 'clicks', onlychanged=False)

        self._widget_pygimli = pn.widgets.Button(name="GeoelectricsModule", button_type="success")
        self._widget_pygimli.param.watch(self._callback_pygimli, 'clicks', onlychanged=False)

        self._widget_devito = pn.widgets.Button(name="SeismicModule", button_type="success")
        self._widget_devito.param.watch(self._callback_devito, 'clicks', onlychanged=False)

        self._widget_torch = pn.widgets.Button(name="LandscapeGeneration", button_type="success")
        self._widget_torch.param.watch(self._callback_torch, 'clicks', onlychanged=False)

        self._widget_calibration_projector = pn.widgets.FileInput(name="Load projector calibration", accept=".json")
        self._widget_calibration_projector.param.watch(self._callback_calibration_projector, 'value')

        self._widget_calibration_sensor = pn.widgets.FileInput(name="Load sensor calibration", accept=".json")
        self._widget_calibration_sensor.param.watch(self._callback_calibration_sensor, 'value')

        self._widget_create_calibration_projector = pn.widgets.Button(name="Calibrate projector", button_type="success")
        self._widget_create_calibration_projector.param.watch(self._callback_create_calibration_projector, 'clicks',
                                                              onlychanged=False)

        self._widget_create_calibration_sensor = pn.widgets.Button(name="Calibrate Sensor", button_type="success")
        self._widget_create_calibration_sensor.param.watch(self._callback_create_calibration_sensor, 'clicks',
                                                           onlychanged=False)

        self._widget_new_server = pn.widgets.Button(name="New Server", button_type="warning")
        self._widget_new_server.param.watch(self._callback_new_server, 'clicks', onlychanged=False)

        self._widget_thread_selector = pn.widgets.RadioButtonGroup(name='Thread controller',
                                                                   options=["Start", "Stop"],
                                                                   value="Start",
                                                                   button_type='success')
        self._widget_thread_selector.param.watch(self._callback_thread_selector, 'value', onlychanged=False)

        self._widget_aruco = pn.widgets.Checkbox(name='Aruco Detection', value=self.ARUCO_ACTIVE,
                                                 disabled=self._disable_aruco)
        self._widget_aruco.param.watch(self._callback_aruco, 'value',
                                       onlychanged=False)

        return True

    def show_widgets(self):
        self._create_widgets()
        widgets = pn.Column("# Module selector",
                            self._widget_main_thread,
                            '<b>Select the module you want to show the widgets and start </b>',
                            self._widget_topo,
                            self._widget_gradient,
                            self._widget_load_save_topo,
                            self._widget_landslide,
                            self._widget_search,
                            '<b>External modules</b>',
                            self._widget_gempy if self.Modules.get('GemPyModule') is not None else None,
                            self._widget_pygimli if self.Modules.get('GeoelectricsModule') is not None else None,
                            self._widget_devito if self.Modules.get('SeismicModule') is not None else None,
                            self._widget_torch if self.Modules.get('LandscapeGeneration') is not None else None,
                            '<b>Change the calibration file</b>',
                            pn.WidgetBox('Projector',
                                         self._widget_calibration_projector,
                                         'Sensor',
                                         self._widget_calibration_sensor),
                            '<b>Create new projector calibration file</b>',
                            self._widget_create_calibration_projector,
                            '<b>Create new sensor calibration file</b>',
                            self._widget_create_calibration_sensor,
                            )
        thread = pn.Column("##<b>Thread Controller</b>",
                           self._widget_thread_selector,
                           '<b>Start a new server</b>',
                           self._widget_new_server,
                           "<b>Deactivate or activate aruco detection</b>",
                           self._widget_aruco,
                           "<b>Manager of all modules</b>",
                           self.Main_Thread._widget_module_selector,
                           self.Main_Thread._widget_clear_axes,
                           self.Main_Thread._widget_error_markdown)

        panel = pn.Row(widgets, thread)

        return panel

    def _callback_main_thread(self, event):
        self.Main_Thread.widget_plot_module().show()

    def _callback_gradient(self, event):
        #self.remove_from_main_thread()
        self.add_to_main_thread('GradientModule')

    def _callback_load_save(self, event):
        #self.remove_from_main_thread()
        self.add_to_main_thread('LoadSaveTopoModule')

    def _callback_topo(self, event):
        #self.remove_from_main_thread()
        self.add_to_main_thread('TopoModule')

    def _callback_landslide(self, event):
        #self.remove_from_main_thread()
        self.add_to_main_thread('LandslideSimulation')

    def _callback_search(self, event):
        #self.remove_from_main_thread()
        self.add_to_main_thread('SearchMethodsModule')

    def _callback_gempy(self, event):
        #self.remove_from_main_thread()
        self.add_to_main_thread('GemPyModule')

    def _callback_pygimli(self, event):
        self.add_to_main_thread('GeoelectricsModule')

    def _callback_devito(self, event):
        self.add_to_main_thread('SeismicModule')

    def _callback_torch(self, event):
        self.add_to_main_thread('LandscapeGeneration')

    def _callback_calibration_projector(self, event):
        global p_width, p_height
        data_bytes = event.new
        data_decode = data_bytes.decode()
        self._projector_calib = data_decode
        self.projector.__init__(data_decode, p_width=p_width, p_height=p_height)
        self.Main_Thread.projector = self.projector

    def _callback_calibration_sensor(self, event):
        global name_sensor
        data_bytes = event.new
        data_decode = data_bytes.decode()
        self._sensor_calib = data_decode
        self.sensor.__init__(data_decode, name=name_sensor)
        self.Main_Thread.sensor = self.sensor

    def _callback_create_calibration_projector(self, event):
        self.projector = calibrate_projector()
        self.Main_Thread.projector = self.projector

    def _callback_create_calibration_sensor(self, event):
        self.Main_Thread.stop()
        self.sensor = calibrate_sensor()
        self.Main_Thread.sensor = self.sensor

    def _callback_new_server(self, event):
        global p_width, p_height
        self.projector.__init__(self._projector_calib, p_width=p_width, p_height=p_height)
        self.Main_Thread.projector = self.projector

    def _callback_thread_selector(self, event):
        if event.new == "Start":
            self.Main_Thread.run()
        elif event.new == "Stop":
            self.Main_Thread.stop()

    def _callback_aruco(self, event):
        self.Main_Thread.ARUCO_ACTIVE = event.new












