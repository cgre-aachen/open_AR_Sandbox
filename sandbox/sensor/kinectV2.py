#from sandbox import _
import numpy
import sys
import platform
import traceback
_platform = platform.system()

try:
    if _platform =='Windows':
        from pykinect2 import PyKinectV2  # Wrapper for KinectV2 Windows SDK
        from pykinect2 import PyKinectRuntime
    elif _platform =='Linux':
        try:
            from pylibfreenect2 import Freenect2, SyncMultiFrameListener
            from pylibfreenect2 import FrameMap, FrameType, Registration, Frame
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
            _pylib=True
        except:
            _pylib=False
            print("No package pylibfreenect2")
        try:
            from freenect2 import Device, FrameType
            _lib = True
        except:
            _lib = False
            print("No package freeenect2")
        if not _pylib and not _lib:
            print('dependencies not found for KinectV2 to work. Check installation and try again')
            raise ImportError
except ImportError:
    print('dependencies not found for KinectV2 to work. Check installation and try again')

_pylib = False #For development change this

class KinectV2:
    """
    control class for the KinectV2 based on the Python wrappers of the official Microsoft SDK
    Init the kinect and provides a method that returns the scanned depth image as numpy array.
    Also we do gaussian blurring to get smoother surfaces.

    """


    def __init__(self):
        # hard coded class attributes for KinectV2's native resolution
        self.name = 'kinect_v2'
        self.depth_width = 512
        self.depth_height = 424
        self.color_width = 1920
        self.color_height = 1080
        self.depth = None
        self.color = None
        self._init_device()

        #self.depth = self.get_frame()
        #self.color = self.get_color()
        print("KinectV2 initialized.")

    def _init_device(self):
        if _platform == 'Windows':
            device = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color |
                                                     PyKinectV2.FrameSourceTypes_Depth |
                                                     PyKinectV2.FrameSourceTypes_Infrared)

        elif _platform == 'Linux':
            if _pylib:
                # self.device = Device()
                fn = Freenect2()
                num_devices = fn.enumerateDevices()
                assert num_devices > 0

                serial = fn.getDeviceSerialNumber(0)
                self.device = fn.openDevice(serial, pipeline=pipeline)

                self.listener = SyncMultiFrameListener(
                    FrameType.Color |
                    FrameType.Ir |
                    FrameType.Depth)

                # Register listeners
                self.device.setColorFrameListener(self.listener)
                self.device.setIrAndDepthFrameListener(self.listener)

                self.device.start()

                self.registration = Registration(self.device.getIrCameraParams(), self.device.getColorCameraParams())

                self.undistorted = Frame(512, 424, 4)
                self.registered = Frame(512, 424, 4)

                frames = self.listener.waitForNewFrame()
                self.listener.release(frames)

                self.device.stop()
                self.device.close()
            elif _lib:
                try:
                    self.device = Device()
                except:
                    traceback.print_exc()

        else:
            print(_platform)
            raise NotImplementedError

    def _get_linux_frame(self, typ:str='all'):
        """
        Manage to
        Args:
            typ:

        Returns:

        """
        #self.device.start()
        frames = self.listener.waitForNewFrame(milliseconds=1000)
        if frames:
            if typ == 'depth':
                depth = frames[FrameType.Depth]
                self.listener.release(frames)
                return depth.asarray()
            elif typ == 'ir':
                ir = frames[FrameType.Ir]
                self.listener.release(frames)
                return ir.asarray()
            if typ == 'color':
                color = frames[FrameType.Color]
                self.listener.release(frames)
                return color.asarray()
            if typ == 'all':
                color = frames[FrameType.Color]
                ir = frames[FrameType.Ir]
                depth = frames[FrameType.Depth]
                self.registration.apply(color, depth, self.undistorted, self.registered)
                self.registration.undistortDepth(depth, self.undistorted)
                self.listener.release(frames)
                return depth.asarray(), color.asarray(), ir.asarray()
        else:
            raise FileNotFoundError

    def get_linux_frame(self, typ:str='all'):
        """
        Manage to
        Args:
            typ:

        Returns:

        """
        frames = {}
        with self.device.running():
            for type_, frame in self.device:
                frames[type_] = frame
                if FrameType.Color in frames and FrameType.Depth in frames and FrameType.Ir in frames:
                    break
        if typ == 'depth':
            depth = frames[FrameType.Depth].to_array()
            return depth
        elif typ == 'ir':
            ir = frames[FrameType.Ir].to_array()
            return ir
        elif typ == 'color':
            color = frames[FrameType.Color].to_array()
            resolution_camera = self.color_height * self.color_width  # resolution camera Kinect V2
            palette = numpy.reshape(color, (resolution_camera, 4))[:, [2, 1, 0]]
            position_palette = numpy.reshape(numpy.arange(0, len(palette), 1), (self.color_height, self.color_width))
            color = palette[position_palette]
            return color
        else:
            color = frames[FrameType.Color].to_array()
            resolution_camera = self.color_height * self.color_width  # resolution camera Kinect V2
            palette = numpy.reshape(color, (resolution_camera, 4))[:, [2, 1, 0]]
            position_palette = numpy.reshape(numpy.arange(0, len(palette), 1), (self.color_height, self.color_width))
            color = palette[position_palette]
            depth = frames[FrameType.Depth].to_array()
            ir = frames[FrameType.Ir].to_array()
            return color, depth, ir

    def get_frame(self):
        """
        Args:
        Returns:
               2D Array of the shape(424, 512) containing the depth information of the latest frame in mm
        """
        if _platform =='Windows':
            depth_flattened = self.device.get_last_depth_frame()
            self.depth = depth_flattened.reshape(
                (self.depth_height, self.depth_width))  # reshape the array to 2D with native resolution of the kinectV2
        elif _platform =='Linux':
            self.depth = self.get_linux_frame(typ='depth')
        return self.depth

    def get_ir_frame_raw(self):
        """
        Args:
        Returns:
               2D Array of the shape(424, 512) containing the raw infrared intensity in (uint16) of the last frame
        """
        if _platform=='Windows':
            ir_flattened = self.device.get_last_infrared_frame()
            self.ir_frame_raw = numpy.flipud(
                ir_flattened.reshape((self.depth_height,
                                      self.depth_width)))  # reshape the array to 2D with native resolution of the kinectV2
        elif _platform=='Linux':
            self.ir_frame_raw = self.get_linux_frame(typ='ir')
        return self.ir_frame_raw

    def get_ir_frame(self, min=0, max=6000):
        """

        Args:
            min: minimum intensity value mapped to uint8 (will become 0) default: 0
            max: maximum intensity value mapped to uint8 (will become 255) default: 6000
        Returns:
               2D Array of the shape(424, 512) containing the infrared intensity between min and max mapped to uint8 of the last frame

        """
        ir_frame_raw = self.get_ir_frame_raw()
        self.ir_frame = numpy.interp(ir_frame_raw, (min, max), (0, 255)).astype('uint8')
        return self.ir_frame

    def get_color(self):
        """

        Returns:

        """
        if _platform == 'Windows':
            color_flattened = self.device.get_last_color_frame()
            resolution_camera = self.color_height * self.color_width  # resolution camera Kinect V2
            # Palette of colors in RGB / Cut of 4th column marked as intensity
            palette = numpy.reshape(numpy.array([color_flattened]), (resolution_camera, 4))[:, [2, 1, 0]]
            position_palette = numpy.reshape(numpy.arange(0, len(palette), 1), (self.color_height, self.color_width))
            self.color = numpy.flipud(palette[position_palette])
        elif _platform =='Linux':
            self.color = self.get_linux_frame(typ='color')
        return self.color
