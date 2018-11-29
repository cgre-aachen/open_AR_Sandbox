#sys.path.append('/home/miguel/PycharmProjects/gempy')
import weakref
import numpy
import scipy
import gempy
import matplotlib.pyplot as plt
from itertools import count
from sandbox.Sandbox import Calibration

#TODO: make calibrations a mandatory argument in ALL Classes!!!




### Still to refactor:

#TODO: use Descriptors
class Model:
    _ids = count(0)
    _instances = []

    def __init__(self, model, calibration=None, lock=None):
        self.id = next(self._ids)
        self.__class__._instances.append(weakref.proxy(self))

        self.legend = True
        self.model = model
        gempy.compute_model(self.model)
        self.empty_depth_grid = None
        self.depth_grid = None

        self.stop_threat = False
        self.lock = lock

        if calibration is None:
            try:
                self.calibration = Calibration._instances[-1]
                print("no calibration specified, using last calibration instance created: ",self.calibration)
            except:
                print("ERROR: no calibration instance found. please create a calibration")
                # parameters from the model:
        else:
            self.calibration = calibration


    def setup(self, start_stream=False):
        if start_stream == True:
            self.calibration.associated_projector.start_stream()
        self.calculate_scales()
        self.create_empty_depth_grid()

    def run(self):
        run_model(self)


## global functions to run the model in loop.
def run_model(model, calibration=None, kinect=None, projector=None, filter_depth=True, n_frames=5,
              sigma_gauss=4):  # continous run functions with exit handling
    if calibration == None:
        calibration = model.calibration
    if kinect == None:
        kinect = calibration.associated_kinect
    if projector == None:
        projector = calibration.associated_projector

    while True:
        if filter_depth == True:
            depth = kinect.get_filtered_frame(n_frames=n_frames, sigma_gauss=sigma_gauss)
        else:
            depth = kinect.get_frame()

        model.update_grid(depth)
        model.render_frame(depth, outfile="current_frame.jpeg")
        #time.sleep(self.delay)
        projector.show(input="current_frame.jpeg", rescale=False)

        if model.stop_threat is True:
            raise Exception('Threat stopped')
