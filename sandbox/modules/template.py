from abc import ABC, abstractmethod


class ModuleTemplate(ABC):
    def __init__(self):
        self.ax = None

    @abstractmethod
    def update(self):
        ### Do all the calculations from the data
        self.plot()
        ### pass the data to the plot to paint in the axes, this will return the axes and a colormap
        pass

    @abstractmethod
    def plot(self):
        pass
