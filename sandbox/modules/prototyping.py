import traceback
from .module_main_thread import Module


class PrototypingModule(Module):
    """
    Class for the connectivity between Notebook plotting and sandbox image in live thread
    """
    def __init__(self, *args, **kwargs):
        # call parents' class init, use greyscale colormap as standard and extreme color labeling
        super().__init__(*args, contours=True, cmap='gist_earth_r', over='k', under='k', **kwargs)

        self.function_to_run = None
        self.active_connection = False

    def setup(self):
        frame = self.sensor.get_frame()
        if self.crop:
            frame = self.crop_frame(frame)
            frame = self.clip_frame(frame)

        self.plot.render_frame(frame)
        self.projector.frame.object = self.plot.figure

    def update(self):
        frame = self.sensor.get_frame()
        if self.crop:
            frame = self.crop_frame(frame)
            frame = self.clip_frame(frame)

        if self.active_connection:
            self.plot.ax.cla()
            try:
                self.function_to_run()
            except Exception:
                traceback.print_exc()
                self.active_connection = False

        else:
            self.plot.render_frame(frame)

        # if aruco Module is specified: update, plot aruco markers
        if self.ARUCO_ACTIVE:
            self.update_aruco()
            self.plot.plot_aruco(self.Aruco.aruco_markers)

        self.projector.trigger()  # triggers the update of the bokeh plot

    def plot_sandbox(self, func):
        """
        Pass as an argument the function to run in the thread
        Args:
            func: see notebook in tutorials fro example

        Returns:

        """
        def inner1(*args, **kwargs):
            frame = self.sensor.get_frame()
            if self.crop:
                frame = self.crop_frame(frame)
                frame = self.clip_frame(frame)
            func(*args, sandbox_ax=self.plot.ax, sandbox_frame=frame,  **kwargs)
        return inner1
