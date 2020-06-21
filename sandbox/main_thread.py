from .sensor.sensor_api import Sensor

class MainThread():
    def update(self, modules: list, **kwargs):
        ax.clear()

        depth = self.sensor.get_frame()
        points = self.aruco.get_loaction()
        for m in modules():
            ax = m.update(**kwargs)

        projector.trigger
        pass

    # TODO All multithreading sutff