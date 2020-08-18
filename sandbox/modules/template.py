from abc import ABC, abstractmethod


class ModuleTemplate(ABC):

    def __init__(self, extent: list = None):
        self.lock = None
        if extent is not None:
            self.vmin = extent[4]
            self.vmax = extent[5]
        pass

    @abstractmethod
    def update(self, sb_params: dict):
        active_cmap = sb_params.get('active_cmap')
        active_contours = sb_params.get('active_contours')
        frame = sb_params.get('frame')
        extent = sb_params.get('extent')
        ax = sb_params.get('ax')
        cmap = sb_params.get('cmap')
        norm = sb_params.get('norm')
        marker = sb_params.get('marker')

        ### Do all the calculations from the data
        self.plot(frame, ax)
        ### pass the data to the plot to paint in the axes, this will return the axes and a colormap

        return sb_params
        pass

    @abstractmethod
    def plot(self, frame, ax):
        pass
