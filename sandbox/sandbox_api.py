# TODO: Load all the modules from here!!! Important for testing (?)
import panel as pn
import traceback
import json
from sandbox import _calibration_dir
# logging and exception handling
verbose = False
if verbose:
    import logging
    logging.basicConfig(filename="main.log",
                        filemode='w',
                        level=logging.WARNING,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        )

def calibrate_projector():
    from sandbox.projector import Projector
    proj = Projector(use_panel=True)
    widget = proj.calibrate_projector()
    widget.show()
    return proj

def calibrate_sensor(filename_projector: str =_calibration_dir + "my_projector_calibration.json", name: str = "kinect_v2"):
    _calibprojector = _calibration_dir + filename_projector
    from sandbox.sensor import CalibSensor
    module = CalibSensor(calibprojector=_calibprojector, name=name)
    widget = module.calibrate_sensor()
    widget.show()
    return module.sensor

def start_server(filename_projector: str =_calibration_dir + "my_projector_calibration.json",
                 filename_sensor: str = _calibration_dir + "my_sensor_calibration.json",
                 sensor_name='kinect_v2',
                 aruco_marker=True,
                 gempy_module=False,
                 **kwargs):
                 #**projector_kwargs,
                 #**sensor_kwargs,
                 #**gempy_kwargs):

    from sandbox.projector import Projector
    from sandbox.sensor import Sensor
    from sandbox.markers import MarkerDetection

    if filename_projector is None:
        projector = Projector(use_panel=True, **kwargs)
    else:
        projector = Projector(calibprojector=filename_projector, use_panel=True, **kwargs)

    if filename_sensor is None:
        sensor = Sensor(name=sensor_name, **kwargs)
    else:
        sensor = Sensor(calibsensor=filename_sensor, name=sensor_name, **kwargs)

    if aruco_marker:
        aruco = MarkerDetection(sensor=sensor)
    else:
        aruco = None


    module = Sandbox(sensor=sensor,
                     projector=projector,
                     aruco=aruco,
                     gempy_module=gempy_module,
                     **kwargs)

    module.start()

    return module


class Sandbox:

    # Wrapping API-class

    def __init__(self, sensor,#: Sensor,
                 projector,#: Projector,
                 aruco,#: MarkerDetection = None,
                 gempy_module: bool = False,
                 **kwargs):


        self.sensor = sensor
        self._sensor_calib = self.sensor.json_filename
        self.projector = projector
        self._projector_calib = self.projector.json_filename
        self.aruco = aruco
        from sandbox.markers import MarkerDetection
        if isinstance(self.aruco, MarkerDetection):
            self.disable_aruco = False
            self.ARUCO_ACTIVE = True
        else:
            self.disable_aruco = True
            self.ARUCO_ACTIVE = False

        self.Modules = None
        self.module_active = False
        self.load_modules(gempy_module, **kwargs)
        self.Main_Thread = self.start_main_thread(**kwargs)

        return print('Sandbox server ready')


    def load_modules(self, gempy_module: bool, **kwargs):
        from sandbox.modules import TopoModule, GradientModule, LoadSaveTopoModule, LandslideSimulation, SearchMethodsModule
        from sandbox.projector import ContourLinesModule, CmapModule
        self.Modules = {}
        self.Modules['ContourLinesModule'] = ContourLinesModule(extent=self.sensor.extent)
        self.Modules['CmapModule'] = CmapModule(extent=self.sensor.extent)
        self.Modules['TopoModule'] = TopoModule(extent=self.sensor.extent)
        self.Modules['GradientModule'] = GradientModule(extent=self.sensor.extent)
        self.Modules['LoadSaveTopoModule'] = LoadSaveTopoModule(extent=self.sensor.extent)
        self.Modules['LandslideSimulation'] = LandslideSimulation(extent=self.sensor.extent)
        self.Modules['SearchMethodsModule'] = SearchMethodsModule(extent=self.sensor.extent)
        self.Modules['SearchMethodsModule'].update_mesh(self.sensor.get_frame(),
                                                        margins_crop=self.Modules['SearchMethodsModule'].margins_crop,
                                                        fill_value=0)
        self.Modules['SearchMethodsModule'].activate_frame_capture = False
        if gempy_module:
            from sandbox.modules import GemPyModule
            geo_model = kwargs.get('geo_model')
            #load_example = kwargs.get('load_examples')
            #name_example = kwargs.get('name_example')
            self.Modules['GemPyModule'] = GemPyModule(geo_model=geo_model,
                                                      extent=self.sensor.extent,
                                                      box=self.sensor.physical_dimensions,
                                                      #load_examples=load_example,
                                                      #name_example=name_example,
                                                      **kwargs)

    def start_main_thread(self, **kwargs):
        from sandbox.main_thread import MainThread
        thread = MainThread(sensor=self.sensor, projector=self.projector, aruco=self.aruco, **kwargs)
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

    def _create_widgets(self):
        self._widget_main_thread = pn.widgets.Button(name="MainThread_Controllers", button_type="success")
        self._widget_main_thread.param.watch(self._callback_main_thread, 'clicks',
                                          onlychanged=False)
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

        self._widget_create_calibration_projector = pn.widgets.Button(name="Calibrate projector", button_type="success")
        self._widget_create_calibration_projector.param.watch(self._callback_create_calibration_projector, 'clicks',
                                                    onlychanged=False)

        self._widget_create_calibration_sensor = pn.widgets.Button(name="Calibrate Sensor", button_type="success")
        self._widget_create_calibration_sensor.param.watch(self._callback_create_calibration_sensor, 'clicks',
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
        if self.Modules.get('GemPyModule') is not None:
            gempy_module = self._widget_gempy
        else:
            gempy_module = None
        widgets = pn.Column("# Module selector",
                            '<b>Select the module you want to show the widgets and start </b>',
                            self._widget_main_thread,
                            self._widget_topo,
                            self._widget_gradient,
                            self._widget_load_save_topo,
                            self._widget_landslide,
                            self._widget_search,
                            gempy_module,
                            '<b>Change the calibration file</b>',
                            self._widget_calibration,
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
                           self.Main_Thread._widget_clear_axes)

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

    def _callback_calibration(self, event):
        data_bytes = event.new
        data_decode = data_bytes.decode()
        data_dict = json.loads(data_decode)
        self.calib.__dict__ = data_dict
        self.sensor.__init__(self.calib)
        self.projector.__init__(self.calib)

    def _callback_create_calibration_projector(self, event):
        self.projector = calibrate_projector()
        self.Main_Thread.projector = self.projector

    def _callback_create_calibration_sensor(self, event):
        self.sensor = calibrate_sensor()
        self.Main_Thread.sensor = self.sensor

    def _callback_new_server(self, event):
        self.projector.__init__(self._projector_calib)
        self.Main_Thread.projector = self.projector

    def _callback_thread_selector(self, event):
        if event.new == "Start":
            self.Main_Thread.run()
        elif event.new == "Stop":
            self.Main_Thread.stop()

    def _callback_aruco(self, event):
        self.Main_Thread.ARUCO_ACTIVE = event.new












