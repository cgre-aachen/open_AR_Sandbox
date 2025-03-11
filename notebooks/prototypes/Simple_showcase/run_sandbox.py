import os,sys
os.chdir(r"C:\Users\chudalla\Desktop\Python_repos\open_AR_Sandbox")
#sys.path.append('../../../')
import panel as pn
#save the paths to the calibration files
from sandbox import _calibration_dir
_calibprojector = _calibration_dir + "my_projector_calibration.json"
_calibsensor = _calibration_dir + "my_sensor_calibration.json"

#Import Sensor
from sandbox.sensor import Sensor
sensor = Sensor(calibsensor=_calibsensor, name="kinect_v2")

# Import projector
from sandbox.projector import Projector
projector = Projector(calibprojector=_calibprojector, use_panel=False)
projector.panel.servable()
from sandbox.main_thread import MainThread
main = MainThread(sensor=sensor, projector=projector)

# Import the modules to use
from sandbox.modules import TopoModule
TopoModule = TopoModule(extent=sensor.extent)


from sandbox.modules.pytorch.landscape_live import GenLandModule 

GenLand = GenLandModule(main.sb_params, resolution=120)
GenLand.restore_model(r"C:\Users\chudalla\Desktop\Python_repos\open_AR_Sandbox\notebooks\tutorials\09_LandscapeGeneration\checkpoints\training_checkpoints")
main.add_module(name="Pix2Pix", module=GenLand)

main.run()#, kwargs={""})


# pn.serve(projector.panel, show=False)
# server.start()
# server.join()
# from sandbox.main_thread import MainThread
# main = MainThread(sensor=sensor, projector=projector)