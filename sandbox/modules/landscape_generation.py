import os
import matplotlib.pyplot as plt
import numpy
import traceback
from sandbox.modules.template import ModuleTemplate
from sandbox.modules import LoadSaveTopoModule
from sandbox import _test_data

class LandscapeGeneration(ModuleTemplate):
    """Class to generate landscapes using DEMs from the sandbox and a pre-trained model with the
    pytorch-CycleGAN-and-pix2pix library"""

    def __init__(self, extent: list = None):
        self.depth_image = None
        self.LoadArea = LoadSaveTopoModule(extent=extent)
        self.show_landscape = False
        self.img = None
        self.DEM = None
        self._img = None
        self.lock = None

    def update(self, sb_params: dict):
        sb_params = self.LoadArea.update(sb_params)
        ax = sb_params.get('ax')
        self.lock = sb_params.get('lock_thread')
        self.plot(ax)

        return sb_params

    def plot(self, ax):
        self.remove_image()
        if self.show_landscape:
            self.image_landscape(ax)

    def remove_image(self):
        if self._img is not None:
            self._img.remove()
            self._img = None

    def get_image_modify(self):
        if self.LoadArea.absolute_topo is None:
            self.DEM, _ = self.LoadArea.extractTopo()
        else:
            self.DEM = self.LoadArea.absolute_topo
        imd_d = numpy.copy(self.DEM)
        DEM = numpy.c_[self.DEM, imd_d]
        return DEM

    def save_image(self, image: numpy.ndarray = None,
                   name: str = 'landscape_image.png',
                   pathname: str = _test_data['landscape_generation']+'saved_DEMs/test/'):
        if image is None:
            image = self.get_image_modify()
        self.lock.acquire()
        fig, ax = plt.subplots()

        ax.imshow(image, cmap='gist_earth', origin="lower left")
        ax.set_axis_off()
        fig.savefig(pathname+name, bbox_inches='tight', pad_inches=0)
        plt.close()
        print("saved succesfully in: " + pathname)
        self.lock.release()
        return fig

    def run_cmd(self,
                package_dir: str,
                dataroot_dir: str = _test_data['landscape_generation']+'saved_DEMs/',
                checkpoints_dir:str = _test_data['landscape_generation']+'checkpoints/',
                results_dir: str = _test_data['landscape_generation']+'results/',
                ):

        to_string = 'python'+' '+os.path.abspath(package_dir)+'/test.py'+' '+\
                    '--dataroot'+' '+os.path.abspath(dataroot_dir)+' '+\
                    '--results_dir'+' '+os.path.abspath(results_dir)+' '+\
                    '--checkpoints_dir'+' '+os.path.abspath(checkpoints_dir)+' '+\
                    '--name train_1k --model pix2pix --gpu_ids -1 --direction AtoB'

        os.popen('call activate sandbox')

        os.popen(to_string)

        print('Landscape generated')
        return to_string

    def read_result(self, name: str = 'landscape_image.png', result_dir: str = _test_data['landscape_generation']+'results\\train_1k\\test_latest\\images'):
        if os.path.isdir(os.path.abspath(result_dir)):
            try:
                self.img = plt.imread(os.path.abspath(result_dir) +'\\'+ name[:-4]+"_fake_B.png")
                print('Image loaded succesfully')
            except Exception:
                traceback.print_exc()
        else:
            print("No image found in %dir", result_dir)

    def image_landscape(self, ax):
        if self.img is not None:
            if self._img is None:
                self._img = ax.imshow(self.img, aspect='auto', extent=self.LoadArea.to_box_extent) #origin='lower left',
            else:
                self._img.set_data(self.img)
        else:
            print("No image to show")


    def show_widgets(self):
        pass