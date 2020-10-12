#%%
import warnings as warn
import matplotlib.pyplot as plt
try:
    from freenect2 import Device, FrameType, lib, ffi, Frame
    FREENECT_INSTALLED = True
except ImportError:
    warn('freenect2 module not found, Coordinate Mapping will not work.')
    FREENECT_INSTALLED = False

try:
    import cv2
    from cv2 import aruco
    CV2_IMPORT = True
except ImportError:
    CV2_IMPORT = False
    warn('opencv is not installed. Object detection will not work')

#%%
from sandbox.sensor.kinectV2 import KinectV2
kinect = KinectV2()

#%%

frames = {}
with kinect.device.running():
    for type_, frame in kinect.device:
        frames[type_] = frame
        if FrameType.Color in frames and FrameType.Depth in frames and FrameType.Ir in frames:
            break
rgb, depth = frames[FrameType.Color], frames[FrameType.Depth]
undistorted_depth, registered_RGB, big_depth = kinect.device.registration.apply(
     rgb, depth, with_big_depth=True )
#%%
points_array = kinect.device.registration.get_points_xyz_array(undistorted_depth)
#%%
plt.imshow(depth.to_array())
plt.show()
plt.imshow(undistorted_depth.to_array())
plt.show()
plt.imshow(registered_RGB.to_array())
plt.show()
plt.imshow(big_depth.to_array())
plt.show()
#%%
points = kinect.device.registration.get_big_points_xyz_array(big_depth)
plt.imshow(points[...,0])
plt.colorbar()
plt.show()
#%%
undistorted = Frame.create(512, 424, 4)
undistorted.format = depth.format
registered = Frame.create(512, 424, 4)
registered.format = rgb.format

big_depth, big_depth_ref = None, ffi.NULL
enable_filter=True
with_big_depth = True
if with_big_depth:
    big_depth = Frame.create(1920, 1082, 4)
    big_depth.format = depth.format
    big_depth_ref = big_depth._c_object

und, reg, big, col_map = lib.freenect2_registration_apply(
    kinect.device.registration._c_object,
    rgb._c_object, depth._c_object, undistorted._c_object,
    registered._c_object, 1 if enable_filter else 0,
    big_depth_ref
)

rvs = [undistorted, registered]
if with_big_depth:
    rvs.append(big_depth)

#%%
lib.freenect2_registration_get_points_xyz()
#%%



