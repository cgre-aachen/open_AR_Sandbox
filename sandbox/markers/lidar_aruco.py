import numpy as np
import pandas as pd
import pyrealsense2 as rs
from sandbox.sensor.lidar_l515 import LiDAR

x_correction = 0
y_correction = 0
#%%
sensor = LiDAR()
#%%

def set_correction(x, y):
    global x_correction, y_correction
    x_correction = x
    y_correction = y

rs.rs2_project_color_pixel_to_depth_pixel()

def start_mapping(kinect: LiDAR):
    """
        Takes the LiDAR L515 sensor and create the map to convert color space to depth space
        Args:
            kinect: Sensor

        Returns:
            pd.Dataframe
        """
    depth_sensor = kinect.profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    align_to = rs.stream.color
    align = rs.align(align_to)
    frames = sensor.pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    #rs.
    # depth_to_color
    #set_device(kinect)
    #set_device_params(kinect)
    #status = kinect._thread_status
    #if status == "running":
    #    kinect._stop()
    #depth, _, undistorted_depth, _, _ = registration()

    #_ = depth_to_camera(undistorted_depth)
    #df = create_CoordinateMap(depth.to_array())
    #print("CoordinateMap created")
    #if status == "running":
    #    kinect._run()
    #return df