import json

class Sandbox():
    # TODO Marco: Object as argument required?

    def __init__(self), calibration_file=None:
        calibration = Calibration(file=calibration_file)
        #projector = Projector(calibration)
        #sensor = Sensor(calibration) --> Adjust for correct child class
        #model = DummyModel()

class Calibration():
    # TODO Marco: Object as argument required?

    def __init__(self, file=None):
        self.p_top_margin = 0
        self.p_left_margin = 0
        self.p_map_width = 0
        self.p_map_height = 0
        # TODO Daniel: Add Kinect calibration parameters

        if file is not None:
            self.load_json(file)

    def load_json(self, file='calibration.json'):
        with open(file) as calibration_json:
            self.__dict__ = json.load(calibration_json)
        print("JSON configuration loaded.")

    def save_json(self, file='calibration.json'):
        with open(file, "w") as calibration_json:
            json.dump(self.__dict__, calibration_json)
        print('JSON configuration file saved:', str(file))

class Projector():













class Sensor():

class Kinect2():
    # child class of Sensor():

class Model():

class DummyModel():
    # child class of Model()

class BlockModel():
    # child class of Model()

class TopoModel():
    # child class of Model()

class GemPyModel():
    # child class of Model()