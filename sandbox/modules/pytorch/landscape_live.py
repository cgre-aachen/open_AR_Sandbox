#from ..template import ModuleTemplate
from sandbox.sensor import Sensor
import os
import shutil
from sandbox.projector import Projector
from sandbox.modules.pytorch.build_model import pix2pix_model
from scipy.ndimage import zoom
from tensorflow import cast, constant, newaxis, float32
import numpy as np
import panel as pn
from sandbox import set_logger, _package_dir
logger = set_logger(__name__)


class GenLandModule:
    def __init__(self, sb_params:dict, resolution:int=256, store_image:bool=False, nframe:int=5):
        """Central class for landscape generation, setting the extent to create landscapes for

        Parameters
        ----------
        sb_params : dict
            Utility dictionary from main class, containing sensor and projector data
        resolution : int, optional
            resolution in points, based on sensor resolution, by default 256
        nframe : int, optional
            Frame to update upon, by default 5
        """

        pn.extension()

        self.tmp_path = "temp_img"
        if not os.path.exists(self.tmp_path):
            os.mkdir(self.tmp_path)
        else:
            for filename in os.listdir(self.tmp_path):
                file_path = os.path.join(self.tmp_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
            print(f"Folder '{self.tmp_path}' already exists. Its content has been deleted.")


        self.store_image = store_image
        self.nframe = nframe
        # Get indices in sensor.depth to select for image generation
        sensor_mid = np.array([int(sb_params["extent"][1]/2), int(sb_params["extent"][3]/2)]) 
        if any(sensor_mid < resolution/2) == True:
            self.transform = True
        else:
            self.transform = False
        
        self.extent = [None, None, None, None]
        
        self.extent[0] = self.min1 = sensor_mid[0] - int(resolution/2)
        self.extent[2] = self.min2 = sensor_mid[1] - int(resolution/2)

        self.extent[1] = self.max1 = sensor_mid[0] + int(resolution/2)
        self.extent[3] = self.max2 = sensor_mid[1] + int(resolution/2)
        
        if self.min1 < 0:
            self.min1 = None
            self.extent[0] = 0
            self.max1 = None
            self.extent[1] = sb_params["extent"][1]
            
        if self.min2 < 0:
            self.min2 = None
            self.extent[2] = 0
            self.max2 = None
            self.extent[3] = sb_params["extent"][3]
        
        self.depth_vals = sb_params["frame"][self.min1:self.max1, self.min2:self.max2]
        scale1, scale2 = self.depth_vals.shape
        self.scale = 256 / np.array([scale1, scale2])
        
        self.img = sb_params["ax"].imshow(np.ones((2,2)), extent=self.extent, origin='lower')
        
        self.count = 0

        logger.info("GenLandModule loaded successfully")
    
    def restore_model(self, filepath:str="checkpoints/training_checkpoints/"):
        """Load a model for landscape generation. Currently loads the most recent training checkpoint from filepath

        Parameters
        ----------
        filepath : str, optional
            Path to checkpoint folder, by default "checkpoints/training_checkpoints/"
        """
        self.model = pix2pix_model()
        self.model.load_model(filepath)
        logger.info("Pix2pix for Landscape Generation succesfully loaded")
        
        
            
    def generate_image(self)->np.ndarray:
        """function to generate landscape of current frame in self.input_vals

        Returns
        -------
        np.ndarray
            Returns RGB image
        """
        img = cast(constant(np.repeat(self.input_vals.reshape((256,256,1)),3,axis=2)), float32) / np.max(self.input_vals)
        return (self.model.generator(img[newaxis, ...], training=True)[0] + 1) / 2
        
        
    def update(self, sb_params:dict)->dict:
        """Update function, in style of all update functions. Generates a landscape on every nframe-th frame

        Parameters
        ----------
        sb_params : dict
            Utility dictionary from main class, containing sensor and projector data
        

        Returns
        -------
        dict
            Returns sb_params

        Raises
        ------
        ValueError
            When model is not loaded, this error is returned. Call restore_model() first and make sure, a folder with checkpoints is selected.
        """
        if self.model.generator == None:
            raise ValueError("Torch model is not loaded, execute **restore_model(filepath)** first.")

        self.count += 1

        if self.count%self.nframe == 0:

        
            self.depth_vals = sb_params["frame"][self.min1:self.max1, self.min2:self.max2]
            self.input_vals = zoom(self.depth_vals, self.scale)

            img_data = self.generate_image().numpy()

            self.img.set_data(img_data)
            
            if self.store_image:
                try:
                #     with  open(self.tmp_path + "/sat_gen.bin", "wb") as file:
                #         img_data.tofile(file)
                #     file.close()

                    d1, d2 = self.depth_vals.shape

                    dm_name = "dem_%i_%i.bin"%(d1,d2)
                #     with open(self.tmp_path + "/"+ dm_name, "wb") as file:
                #         self.depth_vals.tofile(file)
                #     file.close()
                    np.save(self.tmp_path + "/" + dm_name, self.depth_vals)
                    np.save(self.tmp_path + "/sat_gen.npy", img_data)
                except:
                    nothing = 1

            self.count = 1

        return sb_params