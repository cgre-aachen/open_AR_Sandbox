import json

class CalibrationData(object):
    """
        changes from 0.8alpha to 0.9alpha: introduction of box_width and box_height
        changes from 0.9alpha to 1.0alpha: Introduction of aruco corners position
        changes from 1.0alpha to 1.1alpha: Introduction of aruco pose estimation and camera color intrinsic parameters
    """

    def __init__(self,
                 p_width=1280, p_height=800, p_frame_top=0, p_frame_left=0,
                 p_frame_width=600, p_frame_height=450,
                 s_top=10, s_right=10, s_bottom=10, s_left=10, s_min=700, s_max=1500,
                 box_width=1000.0, box_height=800.0,
                 file=None, aruco_corners=None, camera_mtx=None, camera_dist=None):
        """

        Args:
            p_width:
            p_height:
            p_frame_top:
            p_frame_left:
            p_frame_width:
            p_frame_height:
            s_top:
            s_right:
            s_bottom:
            s_left:
            s_min:
            s_max:
            box_width: physical dimensions of the sandbox along x-axis in millimeters
            box_height: physical dimensions of the sandbox along y-axis in millimeters
            aruco_corners: information of the corners if an aruco marker is used
            file:
        """

        # version identifier (will be changed if new calibration parameters are introduced / removed)
        self.version = "1.1alpha"

        # projector
        self.p_width = p_width
        self.p_height = p_height

        self.p_frame_top = p_frame_top
        self.p_frame_left = p_frame_left
        self.p_frame_width = p_frame_width
        self.p_frame_height = p_frame_height

        # self.p_legend_top =
        # self.p_legend_left =
        # self.p_legend_width =
        # self.p_legend_height =

        # hot area
        # self.p_hot_top =
        # self.p_hot_left =
        # self.p_hot_width =
        # self.p_hot_height =

        # profile area
        # self.p_profile_top =
        # self.p_profile_left =
        # self.p_profile_width =
        # self.p_profile_height =

        # sensor (e.g. Kinect)
        self.s_name = 'generic'  # name to identify the associated sensor device
        self.s_width = 500  # will be updated by sensor init
        self.s_height = 400  # will be updated by sensor init

        self.s_top = s_top
        self.s_right = s_right
        self.s_bottom = s_bottom
        self.s_left = s_left
        self.s_min = s_min
        self.s_max = s_max

        self.box_width = box_width
        self.box_height = box_height

        # Aruco Corners
        self.aruco_corners = aruco_corners
        self.camera_mtx = camera_mtx
        self.camera_dist = camera_dist

        if file is not None:
            self.load_json(file)

    # computed parameters for easy access
    @property
    def s_frame_width(self):
        return self.s_width - self.s_left - self.s_right

    @property
    def s_frame_height(self):
        return self.s_height - self.s_top - self.s_bottom

    @property
    def scale_factor(self):
        return (self.p_frame_width / self.s_frame_width), (self.p_frame_height / self.s_frame_height)

    # JSON import/export
    def load_json(self, file):
        with open(file) as calibration_json:
            data = json.load(calibration_json)
            if data['version'] == self.version:
                self.__dict__ = data
                print("JSON configuration loaded.")
            else:
                print("JSON configuration incompatible.\nPlease recalibrate manually!")

    def save_json(self, file='calibration.json'):
        with open(file, "w") as calibration_json:
            json.dump(self.__dict__, calibration_json)
        print('JSON configuration file saved:', str(file))

    def corners_as_json(self, data):
        x = data.to_json()
        self.aruco_corners = x
