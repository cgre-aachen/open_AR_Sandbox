import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
import pyrealsense2 as rs
from sandbox.sensor.lidar_l515 import LiDAR
from sandbox import set_logger
logger = set_logger(__name__)

x_correction = 0
y_correction = 0
depth_min = 0.11  # meter
depth_max = 1.0  # meter
sensor = None
depth_scale = None
depth_intrin = None
color_intrin = None
depth_to_color_extrin = None
color_to_depth_extrin = None
#%%


def set_correction(x, y):
    global x_correction, y_correction
    x_correction = x
    y_correction = y


def set_sensor(lidar):
    global sensor
    sensor = lidar


def set_sensor_params(sensor):
    global depth_scale, depth_intrin, color_intrin, depth_to_color_extrin, color_to_depth_extrin
    depth_scale = sensor.profile.get_device().first_depth_sensor().get_depth_scale()

    depth_intrin = sensor.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    color_intrin = sensor.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    depth_to_color_extrin = sensor.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to(
        sensor.profile.get_stream(rs.stream.color))
    color_to_depth_extrin = sensor.profile.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to(
        sensor.profile.get_stream(rs.stream.depth))


def start_mapping(sensor: LiDAR):
    """
        Takes the LiDAR L515 sensor and create the map to convert color space to depth space
        Args:
            sensor: Sensor

        Returns:
            pd.Dataframe
        """
    set_sensor(sensor)
    set_sensor_params(sensor)
    df = create_CoordinateMap(sensor.get_frame())
    logger.info("CoordinateMap created")
    return df


def project_color_to_depth(xy_pos):
    """
    Be sure to use the set_sensor(...), and set_params(...) before using this method.
    This function will return the coordinate position of from the color frame to the depth frame.
    It uses the rs.rs2_project_color_pixel_to_depth_pixel(...) function
    Args:
        xy_pos:

    Returns:

    """
    while True:
        frames = sensor.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue
        else:
            break
    point = rs.rs2_project_color_pixel_to_depth_pixel(depth_frame.get_data(),
                                                      depth_scale,
                                                      depth_min,
                                                      depth_max,
                                                      depth_intrin,
                                                      color_intrin,
                                                      depth_to_color_extrin,
                                                      color_to_depth_extrin,
                                                      xy_pos)
    return point


def create_CoordinateMap(depth):
    """ Function to create a point to point map of the spatial/pixel equivalences between the depth space, color space and
    camera space. This method requires the depth frame to assign a depth value to the color point.
    Returns:
        CoordinateMap: DataFrame with the x,y,z values of the depth frame; x,y equivalence between the depth space to camera space and
        real world values of x,y and z in meters
    """
    while True:
        frames = sensor.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        else:
            break

    color = np.asanyarray(color_frame.get_data())
    height, width, _ = color.shape
    crop_margin = 5  # When no margi the codo get stuck at the margin value
    x = np.arange(crop_margin, width - crop_margin)
    y = np.arange(crop_margin, height - crop_margin)
    xx, yy = np.meshgrid(x, y)
    xy_points = np.vstack([xx.ravel(), yy.ravel()]).T
    depth_x = []
    depth_y = []
    depth_z = []
    color_x = []
    color_y = []
    # camera_x = [] # TODO: When are the camera values needed?
    # camera_y = []
    # camera_z = []
    for i in tqdm(range(len(xy_points)), desc="Creating CoordinateMap"):
        xcol_point = xy_points[i, 0]
        ycol_point = xy_points[i, 1]
        # if z_point != 0:  # values that do not have depth information cannot be projected to the color space
        point = rs.rs2_project_color_pixel_to_depth_pixel(depth_frame.get_data(),
                                                          depth_scale,
                                                          depth_min,
                                                          depth_max,
                                                          depth_intrin,
                                                          color_intrin,
                                                          depth_to_color_extrin,
                                                          color_to_depth_extrin,
                                                          [xcol_point, ycol_point])
        # If the points are inside the resolution of the depth frame
        if 1 < point[0] < 639 and 1 < point[1] < 479:
            point = list(map(round, point))
            z_point = depth[point[1], point[0]]

            color_x.append(xcol_point)
            color_y.append(ycol_point)
            # TODO: constants added since image is not exact when doing the transformation
            depth_x.append(point[0] + x_correction)
            depth_y.append(point[1] + y_correction)
            depth_z.append(z_point)

    CoordinateMap = pd.DataFrame({'Depth_x': depth_x,
                                  'Depth_y': depth_y,
                                  'Depth_Z(mm)': depth_z,
                                  'Color_x': color_x,
                                  'Color_y': color_y,
                                  # 'Camera_x(m)': camera_x,
                                  # 'Camera_y(m)': camera_y,
                                  # 'Camera_z(m)': camera_z
                                  })
    return CoordinateMap
