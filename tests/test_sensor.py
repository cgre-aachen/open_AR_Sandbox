from sandbox import _test_data
calib_dir = _test_data['test'] + 'temp/'
from sandbox.sensor import Sensor
import numpy as np
import matplotlib.pyplot as plt

#def test_init_kinect_v1():
#    """ Test if detects the kinect 1"""
#    sensor = Sensor(name='kinect_v1')
#    # print(sensor.get_frame(), sensor.get_frame().shape)
#    assert sensor.get_frame().shape == (240, 320)

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
    plt.imshow(sensor.depth, cmap='viridis', origin="lower left")
    plt.show()

def test_linux():
    # An example using startStreams
    import numpy as np
    import cv2
    import sys
    from pylibfreenect2 import Freenect2, SyncMultiFrameListener
    from pylibfreenect2 import FrameType, Registration, Frame

    try:
        from pylibfreenect2 import OpenGLPacketPipeline
        pipeline = OpenGLPacketPipeline()
    except:
        try:
            from pylibfreenect2 import OpenCLPacketPipeline
            pipeline = OpenCLPacketPipeline()
        except:
            from pylibfreenect2 import CpuPacketPipeline
            pipeline = CpuPacketPipeline()
    print("Packet pipeline:", type(pipeline).__name__)

    enable_rgb = False
    enable_depth = True

    fn = Freenect2()
    num_devices = fn.enumerateDevices()
    if num_devices == 0:
        print("No device connected!")
        return None

    serial = fn.getDeviceSerialNumber(0)
    device = fn.openDevice(serial, pipeline=pipeline)

    types = 0
    if enable_rgb:
        types |= FrameType.Color
    if enable_depth:
        types |= (FrameType.Ir | FrameType.Depth)
    listener = SyncMultiFrameListener(types)

    # Register listeners
    device.setColorFrameListener(listener)
    device.setIrAndDepthFrameListener(listener)

    if enable_rgb and enable_depth:
        device.start()
    else:
        device.startStreams(rgb=enable_rgb, depth=enable_depth)

    # NOTE: must be called after device.start()
    if enable_depth:
        registration = Registration(device.getIrCameraParams(),
                                    device.getColorCameraParams())

    undistorted = Frame(512, 424, 4)
    registered = Frame(512, 424, 4)

    def _frame():
        frames = listener.waitForNewFrame()

        if enable_rgb:
            color = frames["color"]
        if enable_depth:
            ir = frames["ir"]
            depth = frames["depth"]

        if enable_rgb and enable_depth:
            registration.apply(color, depth, undistorted, registered)
        elif enable_depth:
            registration.undistortDepth(depth, undistorted)

        if enable_depth:
            plt.imshow(ir.asarray() / 65535.)
            plt.show()
            plt.imshow(depth.asarray() / 4500.)
            plt.show()
            plt.imshow(undistorted.asarray(np.float32) / 4500.)
            plt.show()
        if enable_rgb:
            plt.imshow(cv2.resize(color.asarray(),
                                           (int(1920 / 3), int(1080 / 3))))
            plt.show()
        if enable_rgb and enable_depth:
            plt.imshow(registered.asarray(np.uint8))

        listener.release(frames)

    _frame()
    _frame()
    _frame()
    device.stop()
    device.close()

def test_simplyfy_linux():
    # An example using startStreams
    import numpy as np
    from pylibfreenect2 import Freenect2, SyncMultiFrameListener
    from pylibfreenect2 import FrameType, Registration, Frame

    try:
        from pylibfreenect2 import OpenGLPacketPipeline
        pipeline = OpenGLPacketPipeline()
    except:
        try:
            from pylibfreenect2 import OpenCLPacketPipeline
            pipeline = OpenCLPacketPipeline()
        except:
            from pylibfreenect2 import CpuPacketPipeline
            pipeline = CpuPacketPipeline()
    print("Packet pipeline:", type(pipeline).__name__)

    enable_rgb = False
    enable_depth = True

    fn = Freenect2()
    num_devices = fn.enumerateDevices()
    assert num_devices > 0

    serial = fn.getDeviceSerialNumber(0)
    device = fn.openDevice(serial, pipeline=pipeline)

    listener = SyncMultiFrameListener(
        FrameType.Color | FrameType.Ir | FrameType.Depth)

    # Register listeners
    device.setColorFrameListener(listener)
    device.setIrAndDepthFrameListener(listener)

    device.start()

    registration = Registration(device.getIrCameraParams(), device.getColorCameraParams())

    undistorted = Frame(512, 424, 4)
    registered = Frame(512, 424, 4)

    def _frames():
        frames = listener.waitForNewFrame(milliseconds=1000)

        color = frames[FrameType.Color]
        ir = frames[FrameType.Ir]
        depth = frames[FrameType.Depth]

        registration.apply(color, depth, undistorted, registered)
        registration.undistortDepth(depth, undistorted)

        assert color.width == 1920
        assert color.height == 1080
        assert color.bytes_per_pixel == 4

        assert ir.width == 512
        assert ir.height == 424
        assert ir.bytes_per_pixel == 4

        assert depth.width == 512
        assert depth.height == 424
        assert depth.bytes_per_pixel == 4

        print(color.asarray().shape)
        print(ir.asarray().shape)
        print(depth.asarray().shape)

        listener.release(frames)
    _frames()
    _frames()
    _frames()

def test_init_kinectv2_linux():
    from sandbox.sensor.kinectV2 import KinectV2
    kinect = KinectV2()

def test_get_depth_frame_lx():
    from sandbox.sensor.kinectV2 import KinectV2
    kinect = KinectV2()
    frame = kinect.get_frame()
    print(frame.shape)
    plt.imshow(frame, origin="lower left", cmap="jet")
    plt.colorbar()
    plt.show()
    frame = kinect.get_frame()
    print(frame.shape)
    plt.imshow(frame, origin="lower left", cmap="jet")
    plt.colorbar()
    plt.show()

def test_get_color_frame_lx():
    from sandbox.sensor.kinectV2 import KinectV2
    kinect = KinectV2()
    color = kinect.get_color()
    print(color.shape)
    print(color[0])
    plt.imshow(color, origin="lower left")
    plt.show()
    color = kinect.get_color()
    plt.imshow(color, origin="lower left")
    plt.show()

def test_get_IR_frame_lx():
    from sandbox.sensor.kinectV2 import KinectV2
    kinect = KinectV2()
    IR = kinect.get_ir_frame(min=0, max =6000)
    print(IR.shape)
    plt.imshow(IR, origin="lower left")
    plt.colorbar()
    plt.show()
