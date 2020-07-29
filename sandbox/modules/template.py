from abc import ABC, abstractmethod


class ModuleTemplate(ABC):

    def __init__(self, extent: list = None):
        if extent is not None:
            self.vmin = extent[4]
            self.vmax = extent[5]
        pass

    @abstractmethod
    def update(self, frame, ax, extent, marker, **kwargs):

        cmap = None
        norm = None
        ### Do all the calculations from the data
        self.plot(frame, ax)
        ### pass the data to the plot to paint in the axes, this will return the axes and a colormap

        return frame, ax, extent, cmap, norm
        pass

    @abstractmethod
    def plot(self, frame, ax):
        pass
