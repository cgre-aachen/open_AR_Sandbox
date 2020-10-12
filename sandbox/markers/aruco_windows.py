from warnings import warn
try:
    from pykinect2 import PyKinectV2  # Wrapper for KinectV2 Windows SDK
    from pykinect2 import PyKinectRuntime
    PYKINECT_INSTALLED = True
except ImportError:
    warn('pykinect2 module not found, Coordinate Mapping will not work.')
    PYKINECT_INSTALLED = False

try:
    import cv2
    from cv2 import aruco
    CV2_IMPORT = True
except ImportError:
    CV2_IMPORT = False
    warn('opencv is not installed. Object detection will not work')

#%%
import numpy as np
import cv2
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel

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

# Create and set logger
logger = createConsoleLogger(LoggerLevel.Debug)
setGlobalLogger(logger)

fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print("No device connected!")
    sys.exit(1)

serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial, pipeline=pipeline)

listener = SyncMultiFrameListener(
    FrameType.Color | FrameType.Ir | FrameType.Depth)

# Register listeners
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)

device.start()

# NOTE: must be called after device.start()
registration = Registration(device.getIrCameraParams(),
                            device.getColorCameraParams())

undistorted = Frame(512, 424, 4)
registered = Frame(512, 424, 4)

# Optinal parameters for registration
# set True if you need
need_bigdepth = True
need_color_depth_map = True

bigdepth = Frame(1920, 1082, 4) if need_bigdepth else None
color_depth_map = np.zeros((424, 512),  np.int32).ravel() \
    if need_color_depth_map else None
#%%
frames = listener.waitForNewFrame()

color = frames[FrameType.Color]
ir = frames[FrameType.Ir]
depth = frames[FrameType.Depth]

registration.apply(color, depth, undistorted, registered,
                   bigdepth=bigdepth,
                   color_depth_map=color_depth_map)



