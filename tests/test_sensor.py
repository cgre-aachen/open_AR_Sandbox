import sandbox as sb
import numpy as np


def test_init_kinect_v2():
    """Test it detects the kinect 2"""

    # This line should not be necessary
    calibration = sb.CalibrationData()

    sensor = sb.Sensor(calibration, name='kinect_v2')
    # print(sensor.get_frame(), sensor.get_frame().shape)
    assert sensor.get_frame().shape == (424, 512)

    sensor = sb.Sensor(calibration, name='dummy', random_seed=1234)
    print(sensor.depth[0, 0],
          sensor.depth[0, 0],
          sensor.depth[0, 0])

    assert sensor.depth[0,0] == 1318.183752406038


def test_load_sensor_calibration():
    pass


def test_properties():
    pass

