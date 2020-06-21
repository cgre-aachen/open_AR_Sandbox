#%%
import sandbox as sb

def test_init_sandbox():
    calib = sb.CalibrationData()
    sensor = sb.Sensor(calib, name = 'dummy')
    projector = sb.Projector(calib)

    module = sb.Sandbox(calib, sensor, projector)

    module.show_widgets().show()
