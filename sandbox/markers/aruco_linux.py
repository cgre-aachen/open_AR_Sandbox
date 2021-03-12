#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
from sandbox.sensor.kinectV2 import KinectV2

#%%
depth_params = None
color_params = None
distorted_depth = None
undistorted_depth = None
registered_RGB = None
big_depth = None
device = None
dp_camera_x = None
dp_camera_y = None
dp_camera_z = None
rgb = None
depth = None
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

def set_device_params(kinect):
    global depth_params, color_params
    depth_params = kinect.device.ir_camera_params
    color_params = kinect.device.color_camera_params

def start_mapping(kinect: KinectV2):
    """
        Takes the kinect sensor and create the map to convert color space to depth space
        Args:
            kinect: Sensor

        Returns:
            pd.Dataframe
        """
    # depth_to_color
    set_device(kinect)
    set_device_params(kinect)
    status = kinect._thread_status
    if status == "running":
        kinect._stop()
    depth, _, undistorted_depth, _, _ = registration()

    _ = depth_to_camera(undistorted_depth)
    df = create_CoordinateMap(depth.to_array())
    print("CoordinateMap created")
    if status == "running":
        kinect._run()
    return df

def registration():
    """
    Takes an original frame from the freenect library to be able to create the undistorted depth frame
    Returns:
        undistorted_depth, registered_RGB, big_depth
    """
    from freenect2 import FrameType
    global distorted_depth

    frames = {}
    with device.running():
        for type_, frame in device:
            frames[type_] = frame
            if FrameType.Color in frames and FrameType.Depth in frames and FrameType.Ir in frames:
                break
    global rgb, depth
    rgb, depth = frames[FrameType.Color], frames[FrameType.Depth]

    global undistorted_depth, registered_RGB, big_depth
    undistorted_depth, registered_RGB, big_depth = device.registration.apply(
        rgb, depth, with_big_depth=True)
    return depth, rgb, undistorted_depth, registered_RGB, big_depth

def distort(mx, my):
    """
    see http://en.wikipedia.org/wiki/Distortion_(optics) for description
    """
    dx = (mx - depth_params.cx) / depth_params.fx
    dy = (my - depth_params.cy) / depth_params.fy
    dx2 = dx * dx
    dy2 = dy * dy
    r2 = dx2 + dy2
    dxdy2 = 2 * dx * dy
    kr = 1 + ((depth_params.k3 * r2 + depth_params.k2) * r2 + depth_params.k1) * r2
    x = depth_params.fx * (dx * kr + depth_params.p2 * (r2 + 2 * dx2) + depth_params.p1 * dxdy2) + depth_params.cx;
    y = depth_params.fy * (dy * kr + depth_params.p1 * (r2 + 2 * dy2) + depth_params.p2 * dxdy2) + depth_params.cy;
    return x, y

def depth_to_color(mx, my, z):
    """
    Takes the indexes mx, my from the depth space -and the value of depth at those indexes, to get the
    index position in the color space
    Args:
        mx: index in x position
        my: index in y position
        z: z[my][mx]

    Returns:
        rx, ry
    """
    # these seem to be hardcoded in the original SDK static const float
    depth_q = 0.01
    color_q = 0.002199

    mx = (mx - depth_params.cx) * depth_q
    my = (my - depth_params.cy) * depth_q

    wx = (mx * mx * mx * color_params.mx_x3y0) + \
        (my * my * my * color_params.mx_x0y3) + \
        (mx * mx * my * color_params.mx_x2y1) + \
         (my * my * mx * color_params.mx_x1y2) + \
        (mx * mx * color_params.mx_x2y0) + \
         (my * my * color_params.mx_x0y2) + \
         (mx * my * color_params.mx_x1y1) + \
         (mx * color_params.mx_x1y0) + \
         (my * color_params.mx_x0y1) + \
         (color_params.mx_x0y0)

    wy = (mx * mx * mx * color_params.my_x3y0) + \
         (my * my * my * color_params.my_x0y3) + \
        (mx * mx * my * color_params.my_x2y1) + \
         (my * my * mx * color_params.my_x1y2) + \
        (mx * mx * color_params.my_x2y0) + \
         (my * my * color_params.my_x0y2) + \
         (mx * my * color_params.my_x1y1) + \
        (mx * color_params.my_x1y0) + \
         (my * color_params.my_x0y1) + \
         (color_params.my_x0y0)

    rx = (wx / (color_params.fx * color_q)) - (color_params.shift_m / color_params.shift_d)
    ry = (wy / color_q) + color_params.cy
    #calculating x offset for rgb image based on depth value
    rx = (rx + (color_params.shift_m / z)) * color_params.fx + (color_params.cx+0.5)

    return rx, ry

def depth_to_camera(depth=None):
    """
    Gets the real space coordinates in [m], from the focal point of the camera
    Args:
        depth: undistorted_depth
    Returns:

    """
    if depth is None:
        undistorted = depth
    else:
        undistorted = undistorted_depth

    global dp_camera_x, dp_camera_y, dp_camera_z
    points = device.registration.get_points_xyz_array(undistorted=undistorted)
    dp_camera_x, dp_camera_y, dp_camera_z = points[..., 0], points[..., 1], points[..., 2]
    return dp_camera_x, dp_camera_y, dp_camera_z

def create_CoordinateMap(depth):
    """ Function to create a point to point map of the spatial/pixel equivalences between the depth space, color space and
    camera space. This method requires the depth frame to assign a depth value to the color point.
    Returns:
        CoordinateMap: DataFrame with the x,y,z values of the depth frame; x,y equivalence between the depth space to camera space and
        real world values of x,y and z in meters
    """
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
        if z_point != 0:  # values that do not have depth information cannot be projected to the color space
            x, y = depth_to_color(x_point, y_point, z_point)

            camx = dp_camera_x[y_point][x_point]
            camy = dp_camera_y[y_point][x_point]
            camz = dp_camera_z[y_point][x_point]
            # since the position of the camera and sensor are different, they will not have the same coverage.
            # Specially in the extremes
            if y > 0:

                depth_x.append(x_point)
                depth_y.append(y_point)
                depth_z.append(z_point)
                camera_x.append(camx)
                camera_y.append(camy)
                camera_z.append(camz)
                ####TODO: constants addded since image is not exact when doing the transformation
                color_x.append(round(x) + x_correction)
                color_y.append(round(y) + y_correction)

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


def plot_all(df=None, index=7000):
    """plot all the frames from the script"""
    plt.imshow(depth.to_array(), origin='lower')
    plt.colorbar()
    plt.show()
    plt.imshow(undistorted_depth.to_array(), origin='lower')
    plt.colorbar()
    plt.show()
    plt.imshow(registered_RGB.to_array(), origin='lower')
    plt.colorbar()
    plt.show()
    plt.imshow(big_depth.to_array(), origin='lower')
    plt.colorbar()
    plt.show()

    if df is not None:
        df_ind = df.iloc[index]
        plt.imshow(depth.to_array(), origin='lower')
        plt.plot(df_ind.Depth_x, df_ind.Depth_y, '*r')
        plt.colorbar()
        plt.show()

        plt.imshow(rgb.to_array(), origin='lower')
        plt.plot(df_ind.Color_x, df_ind.Color_y, '*r')
        plt.colorbar()
        plt.show()
#%%
#kinect = KinectV2()
#%%
#df = start_mapping(kinect)
#%%
#plot_all(df, index = 7000)
#%%
