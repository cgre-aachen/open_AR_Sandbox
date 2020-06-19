from abc import ABC, abstractmethod



class ModuleTemplate(ABC):
    def __init__(self):

        self.values = None

    @abstractmethod
    def update(self):


        self.plot()
        pass

    @abstractmethod
    def plot(self):

        pass