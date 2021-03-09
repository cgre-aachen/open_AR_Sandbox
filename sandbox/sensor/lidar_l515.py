import numpy as np
import threading

# %%
try:
    import pyrealsense2 as rs
except ImportError:
    print('dependencies not found for LiDAR L515 to work. Check installation and try again')


# %%
class LiDAR:
    """
    control class for the LiDAR L515 based on the Python wrappers of the Intel RealSense.
    Init the LiDAR and provides a method that returns the scanned depth image as numpy array.
    Also we do gaussian blurring to get smoother surfaces.

    """

    def __init__(self):
        # hard coded class attributes for KinectV2's native resolution
        self.name = 'lidar'
        self.depth_width = 640
        self.depth_height = 480
        self.color_width = 960
        self.color_height = 540

        self._init_device()

        self.depth = self.get_frame()
        self.color = self.get_color()
        print("LiDAR initialized.")

    def _init_device(self):
        """
        Open the pipeline for adquiring frames
        Returns:

        """

        # Threading
        self._lock = threading.Lock()
        self._thread = None
        self._thread_status = 'stopped'  # status: 'stopped', 'running'

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 30)
        if device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.rgb8, 30)
        else:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

        # Start streaming
        self.profile = self.pipeline.start(self.config)

        self._color = np.zeros((self.color_height, self.color_width, 3))
        self._depth = np.zeros((self.depth_height, self.depth_width))
        self._ir = np.zeros((self.depth_height, self.depth_width))
        self._run()
        print("Searching first frame")
        while True:
            if not np.all(self._depth == 0):
                print("First frame found")
                break

    def _run(self):
        """
        Run the thread when _platform is linux
        """
        if self._thread_status != 'running':
            self._thread_status = 'running'
            self._thread = threading.Thread(target=self._update_frames, daemon=True, )
            self._thread.start()
            print('Acquiring frames...')
        else:
            print('Already running.')

    def _stop(self):
        """
        Stop the thread
        """
        if self._thread_status is not 'stopped':
            self._thread_status = 'stopped'  # set flag to end thread loop
            self._thread.join()  # wait for the thread to finish
            print('Stopping frame acquisition.')
        else:
            print('thread was not running.')

    def _update_frames(self):
        """
        keep the stream open to acquire new frames
        """
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            infrared_frame = frames.get_infrared_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame or not infrared_frame:
                continue

            # Convert images to numpy arrays
            self._depth = np.asanyarray(depth_frame.get_data())
            self._color = np.asanyarray(color_frame.get_data())
            self._ir = np.asanyarray(infrared_frame.get_data())

            if self._thread_status != "running":
                break

    def get_frame(self):
        """
        Get depth frame
        Args:
        Returns:
               2D Array of the shape(480, 640) containing the depth information of the latest frame in mm
        """
        self.depth = self._depth
        return self.depth

    def get_ir_frame_raw(self):
        """
        Args:
        Returns:
               2D Array of the shape(480, 640) containing the raw infrared intensity in (uint16) of the last frame
        """
        return self._ir

    def get_ir_frame(self, vmin=0, vmax=6000):
        """

        Args:
            min: minimum intensity value mapped to uint8 (will become 0) default: 0
            max: maximum intensity value mapped to uint8 (will become 255) default: 6000
        Returns:
               2D Array of the shape(480, 640) containing the infrared intensity between min and max mapped to uint8
               of the last frame
        """
        ir_frame_raw = self.get_ir_frame_raw()
        self.ir_frame = np.interp(ir_frame_raw, (vmin, vmax), (0, 255)).astype('uint8')
        return self.ir_frame

    def get_color(self):
        """
        Get the RGB image
        Returns:

        """
        self.color = self._color
        return self.color
