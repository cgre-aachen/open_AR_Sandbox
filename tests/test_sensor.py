import sandbox as sb
calib_dir = sb._calibration_dir

def test_init_kinect_v1():
    #TODO: check with kinect connected
    pass

def test_init_kinect_v2():
    """Test if detects the kinect 2"""
    sensor = sb.Sensor(name='kinect_v2')
    # print(sensor.get_frame(), sensor.get_frame().shape)
    assert sensor.get_frame().shape == (424, 512)

def test_init_dummy():
    sensor = sb.Sensor(name='dummy', random_seed=1234)
    print(sensor.depth[0, 0],
          sensor.depth[0, 0],
          sensor.depth[0, 0])
    assert sensor.depth[0, 0] == 1318.183752406038

def test_save_load_calibration_projector():
    sensor = sb.Sensor()
    file = calib_dir + 'test_sensor_calibration.json'
    sensor.save_json(file=file)
    # now to test if it loads correctly the saved one
    sensor2 = sb.Sensor(calibsensor=file)

def test_get_frame_croped_clipped():
    sensor = sb.Sensor(crop_values=True, clip_values=True)
    print(sensor.get_frame())


