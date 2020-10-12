from sandbox import _test_data
calib_dir = _test_data['test'] + 'temp/'
from sandbox.sensor import Sensor
import numpy as np
import matplotlib.pyplot as plt
import platform
_platform = platform.system()

def test_init_kinect_v1():
    """ Test if detects the kinect 1"""

    sensor = Sensor(name='kinect_v1')
    print(sensor.get_frame(), sensor.get_frame().shape)
    assert sensor.get_frame().shape == (240, 320)

def test_init_kinect_v2():
    """Test if detects the kinect 2"""
    sensor = Sensor(name='kinect_v2', crop_values = False)
    # print(sensor.get_frame(), sensor.get_frame().shape)
    assert sensor.get_frame().shape == (424, 512)

def test_init_dummy():
    sensor = Sensor(name='dummy', random_seed=1234)
    print(sensor.depth[0, 0],
          sensor.depth[0, 0],
          sensor.depth[0, 0])
    #assert np.allclose(sensor.depth[0, 0], 1314.7485240531175)

def test_save_load_calibration_projector():
    sensor = Sensor(name='dummy')
    file = calib_dir + 'test_sensor_calibration.json'
    sensor.save_json(file=file)
    # now to test if it loads correctly the saved one
    sensor2 = Sensor(name='dummy', calibsensor=file)

def test_get_frame_croped_clipped():
    sensor = Sensor(name='dummy', crop_values=True, clip_values=True)
    frame = sensor.get_frame()
    print(frame.shape,  frame)
    assert frame.shape == (404, 492)

def test_extent_property():
    sensor = Sensor(name='dummy')
    print(sensor.extent)
    assert np.allclose(np.asarray([0, 492, 0, 404, 0, 800]), sensor.extent)

def test_get_frame():
    sensor = Sensor(name='kinect_v2', invert=False)
    print(sensor.get_frame())
    plt.imshow(sensor.depth, cmap='viridis', origin="lower")
    plt.show()

def test_init_kinectv2_linux():
    from sandbox.sensor.kinectV2 import KinectV2
    kinect = KinectV2()

def test_kinectv2_linux_frame():
    from sandbox.sensor.kinectV2 import KinectV2
    kinect = KinectV2()
    kinect.get_linux_frame(typ="all")

def test_get_depth_frame_lx():
    from sandbox.sensor.kinectV2 import KinectV2
    kinect = KinectV2()
    frame = kinect.get_frame()
    print(frame.shape)
    plt.imshow(frame, origin="lower", cmap="jet")
    plt.colorbar()
    plt.show()
    frame = kinect.get_frame()
    print(frame.shape)
    plt.imshow(frame, origin="lower", cmap="jet")
    plt.colorbar()
    plt.show()

def test_get_color_frame_lx():
    from sandbox.sensor.kinectV2 import KinectV2
    kinect = KinectV2()
    color = kinect.get_color()
    print(color.shape)
    print(color[0])
    plt.imshow(color, origin="lower")
    plt.show()
    color = kinect.get_color()
    plt.imshow(color, origin="lower")
    plt.show()

def test_get_IR_frame_lx():
    from sandbox.sensor.kinectV2 import KinectV2
    kinect = KinectV2()
    IR = kinect.get_ir_frame(min=0, max =6000)
    print(IR.shape)
    plt.imshow(IR, origin="lower")
    plt.colorbar()
    plt.show()
    IR = kinect.get_ir_frame(min=0, max=6000)
    print(IR.shape)
    plt.imshow(IR, origin="lower")
    plt.colorbar()
    plt.show()

def test_linux_2():
    if _platform == 'Windows':
        return
    from freenect2 import Device, FrameType

    # We use numpy to process the raw IR frame
    import numpy as np

    # We use the Pillow library for saving the captured image
    from PIL import Image

    # Open default device
    device = Device()

    # Start the device
    with device.running():
        # For each received frame...
        for type_, frame in device:
            # ...stop only when we get an IR frame
            if type_ is FrameType.Ir:
                break

    # Outside of the 'with' block, the device has been stopped again

    # The received IR frame is in the range 0 -> 65535. Normalise the
    # range to 0 -> 1 and take square root as a simple form of gamma
    # correction.
    ir_image = frame.to_array()
    ir_image /= ir_image.max()
    ir_image = np.sqrt(ir_image)

def test_linux_2_1():
    if _platform == 'Windows':
        return
    from freenect2 import Device, FrameType
    import numpy as np

    # Open the default device and capture a color and depth frame.
    device = Device()
    frames = {}
    with device.running():
        for type_, frame in device:
            frames[type_] = frame
            if FrameType.Color in frames and FrameType.Depth in frames:
                break

    # Use the factory calibration to undistort the depth frame and register the RGB
    # frame onto it.
    rgb, depth = frames[FrameType.Color], frames[FrameType.Depth]
    undistorted, registered, big_depth = device.registration.apply(
        rgb, depth, with_big_depth=True)

def test_thread_linux():
    from sandbox.sensor.kinectV2 import KinectV2
    kinect = KinectV2()

    kinect._run()
    import time
    time.sleep(1)
    print(kinect._depth)
    print(kinect._ir)
    print(kinect._color)

    kinect._stop()

    kinect._run()
    time.sleep(1)
    print(kinect._depth)
    print(kinect._ir)
    print(kinect._color)

    print("working", kinect.get_color())

def test_thread():
    from sandbox.sensor.kinectV2 import KinectV2
    kinect = KinectV2()
    kinect._run()
    rval = True
    import cv2
    while rval:
        cv2.imshow("color", kinect.get_color())
        cv2.imshow("depth", kinect.get_frame())
        cv2.imshow("ir", kinect.get_ir_frame())
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

def test_thread_linux():
    from sandbox.sensor.kinectV2 import KinectV2
    kinect = KinectV2()