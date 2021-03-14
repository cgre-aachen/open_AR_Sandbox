import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
from sandbox.sensor.kinectV2 import KinectV2

#%%
device = None
x_correction = 0
y_correction = 0

#%%
def set_correction(x, y):
    global x_correction, y_correction
    x_correction = x
    y_correction = y

def set_device(kinect):
    global device
    device = kinect.device

def start_mapping(kinect: KinectV2):
    """
    Takes the kinect sensor and create the map to convert color space to depth space
    Args:
        kinect: Sensor

    Returns:
        pd.Dataframe
    """
    set_device(kinect)
    df = create_CoordinateMap(kinect.get_frame())
    print("CoordinateMap created")
    return df

def create_CoordinateMap(depth):
    """ Function to create a point to point map of the spatial/pixel equivalences between the depth space, color space and
    camera space. This method requires the depth frame to assign a depth value to the color point.
    Returns:
        CoordinateMap: DataFrame with the x,y,z values of the depth frame; x,y equivalence between the depth space to camera space and
        real world values of x,y and z in meters
    """
    from pykinect2 import PyKinectV2
    height, width = depth.shape
    x = np.arange(0, width)
    y = np.arange(0, height)
    xx, yy = np.meshgrid(x, y)
    xy_points = np.vstack([xx.ravel(), yy.ravel()]).T
    depth_x = []
    depth_y = []
    depth_z = []
    camera_x = []
    camera_y = []
    camera_z = []
    color_x = []
    color_y = []
    for i in tqdm(range(len(xy_points)), desc="Creating CoordinateMap"):
        x_point = xy_points[i, 0]
        y_point = xy_points[i, 1]
        z_point = depth[y_point][x_point]
        if z_point != 0:   # values that do not have depth information cannot be projected to the color space
            point = PyKinectV2._DepthSpacePoint(x_point, y_point)
            col = device._mapper.MapDepthPointToColorSpace(point, z_point)
            cam = device._mapper.MapDepthPointToCameraSpace(point, z_point)
            # since the position of the camera and sensor are different, they will not have the same coverage. Specially in the extremes
            if col.y > 0:
                depth_x.append(x_point)
                depth_y.append(y_point)
                depth_z.append(z_point)
                camera_x.append(cam.x)
                camera_y.append(cam.y)
                camera_z.append(cam.z)
                ####TODO: constants addded since image is not exact when doing the transformation
                color_x.append(round(col.x) + x_correction)
                color_y.append(round(col.y) + y_correction)

    CoordinateMap = pd.DataFrame({'Depth_x': depth_x,
                                  'Depth_y': depth_y,
                                  'Depth_Z(mm)': depth_z,
                                  'Color_x': color_x,
                                  'Color_y': color_y,
                                  'Camera_x(m)': camera_x,
                                  'Camera_y(m)': camera_y,
                                  'Camera_z(m)': camera_z
                                  })

    return CoordinateMap



