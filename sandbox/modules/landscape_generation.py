import os
import time
import matplotlib.pyplot as plt
import numpy
import traceback
from sandbox.modules.template import ModuleTemplate
from sandbox.modules import LoadSaveTopoModule
from sandbox import _test_data
try:
    import torch
except ImportError as e:
    traceback.print_exception(e)


class LandscapeGeneration(ModuleTemplate):
    """Class to generate landscapes using DEMs from the sandbox and a pre-trained model with the
    pytorch-CycleGAN-and-pix2pix library"""

    def __init__(self, extent: list = None, package_dir: str = None):
        self.depth_image = None
        self.LoadArea = LoadSaveTopoModule(extent=extent)
        self.show_landscape = False
        self.img = None
        self.DEM = None
        self._img = None
        self.lock = None
        self.last_modified = None
        self.package_dir = package_dir
        self.live_update = True
        #DEM = self.get_image_modify()
        #fig = self.save_image(DEM)
        #self.run_cmd(self.package_dir)

    def update(self, sb_params: dict):
        sb_params = self.LoadArea.update(sb_params)
        ax = sb_params.get('ax')
        self.lock = sb_params.get('lock_thread')
        change = sb_params.get('same_frame')
        #TODO: Include live update of the image
        #if not change and self.live_update:
        #    self.update_model()
        self.plot(ax)

        return sb_params

    def plot(self, ax):
        self.remove_image()
        if self.show_landscape:
            self.image_landscape(ax)

    def update_model(self):
        result_file = _test_data['landscape_generation'] + 'results\\train_1k\\test_latest\\images\\landscape_image_fake_B.png'
        DEM = self.get_image_modify()
        fig = self.save_image(DEM)
        self.run_cmd(self.package_dir)
        if self.last_modified is None:
            self.read_result()
        if self.last_modified != time.ctime(os.path.getmtime(result_file)):
            self.read_result()
        self.last_modified = time.ctime(os.path.getmtime(result_file))

        return print("model updated")

    def set_package_dir(self, package_dir):
        self.package_dir = package_dir

    def remove_image(self):
        """For each frame we need to clear the loaded image so if we change position size or the image
        itself then it will display the most actual one without occupying memory """
        if self._img is not None:
            self._img.remove()
            self._img = None

    def get_image_modify(self):
        """From the LoadSaveTopoModule acquire the dem. If a frame have not been yet loaded then capture a new frame.
        at today 27/08/2020 the image must be doubled for the method to work"""
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
        """
        Takes the image and saves it as a .png 'image' in the 'patchname' folder with 'name' as name
        Args:
            image: Takes a numpy array to be saved as an image. If none then it gets a new image from self.get_image_modify
            name: name of the image. Must include the .png extension
            pathname: location of the image to be saved

        Returns:
            the figure that will be saved
        """
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
                package_dir: str = None,
                dataroot_dir: str = _test_data['landscape_generation']+'saved_DEMs/',
                checkpoints_dir: str = _test_data['landscape_generation']+'checkpoints/',
                results_dir: str = _test_data['landscape_generation']+'results/',
                cmd_string=None):
        """
        Construct the string that will be run in the command line command.
        Args:
            package_dir: The location of the pytorch-CycleGAN-and-pix2pix folder
            dataroot_dir: The location of the image
            checkpoints_dir: The location of the trained model
            results_dir: The location where the results will be saved
            cmd_string: If not None then it will run this string instead of the previous arguments

        Returns:

        """
        if package_dir is None:
            package_dir = self.package_dir

        to_string = 'python'+' '+os.path.abspath(package_dir)+'/test.py'+' '+\
                    '--dataroot'+' '+os.path.abspath(dataroot_dir)+' '+\
                    '--results_dir'+' '+os.path.abspath(results_dir)+' '+\
                    '--checkpoints_dir'+' '+os.path.abspath(checkpoints_dir)+' '+\
                    '--name train_1k --model pix2pix --gpu_ids -1 --direction AtoB'

        os.popen('call activate sandbox')
        if cmd_string is None:
            os.popen(to_string)
        else:
            os.popen(cmd_string)

        print('Landscape generated')
        return to_string

    def read_result(self, name: str = 'landscape_image.png',
                    result_dir: str = _test_data['landscape_generation']+'results\\train_1k\\test_latest\\images'):
        """
        Read the result image from the self.run_cmd(*args) function. It reads the results from the
        'result_dir' that have name 'name'. Be sure to include the .png extension
        Args:
            name: name of image input (.png)
            result_dir: folder of results

        Returns:

        """
        if os.path.isdir(os.path.abspath(result_dir)):
            try:
                file = os.path.abspath(result_dir) +'\\'+ name[:-4]+"_fake_B.png"
                self.img = plt.imread(file)
                if self.last_modified is None:
                    self.last_modified = time.ctime(os.path.getmtime(file))
                print('Image loaded succesfully')
            except Exception:
                traceback.print_exc()
        else:
            print("No image found in %s" %result_dir)

    def image_landscape(self, ax):
        """
        Show the loaded image in the sandbox
        Args:
            ax: axes of the sandbox

        Returns:

        """

        if self.DEM is not None:
            if self._img is None:
                self._img = ax.imshow(self.img, aspect='auto', extent=self.LoadArea.to_box_extent) #origin='lower left',
            else:
                self._img.set_data(self.img)
        else:
            print("No DEM image to show")


    def show_widgets(self):
        pass