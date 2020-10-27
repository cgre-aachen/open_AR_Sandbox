from sandbox import _test_data
from sandbox.modules import LandscapeGeneration
import matplotlib.pyplot as plt
import pytest

import numpy as np
file = np.load(_test_data['topo'] + "DEM1.npz")
frame = file['arr_0']
extent = [0, frame.shape[1], 0, frame.shape[0], frame.min(), frame.max()]

fig, ax = plt.subplots()
pytest.sb_params = {'frame': frame,
                    'ax': ax,
                    'fig': fig,
                    'extent': extent,
                    'marker': [],
                    'cmap': plt.cm.get_cmap('viridis'),
                    'norm': None,
                    'active_cmap': True,
                    'active_contours': True}

def test_init():
    module = LandscapeGeneration()
    print(module)

def test_run_cmd():
    import subprocess
    import os
    os.chdir('C:\\Users\\Admin\\PycharmProjects\\pytorch-CycleGAN-and-pix2pix')
    os.popen('python test.py --dataroot ./datasets/ --name train_1k --model pix2pix --gpu_ids -1 --direction AtoB')
    #list_files = subprocess.run(['ls', '-l'])
    #print(list_files)

def test_run_():
    import os, sys
    package_dir = 'C:\\Users\\Admin\\PycharmProjects\\pytorch-CycleGAN-and-pix2pix\\'
    sys.path.append(package_dir)

    import os
    from options.test_options import TestOptions, BaseOptions
    from data import create_dataset
    from models import create_model
    from util.visualizer import save_images
    from util import html
    opt = TestOptions()

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #print(parser)
    #parser.convert_arg_line_to_args()

    bas = BaseOptions()
    parser = bas.initialize(parser)
    dataroot = _test_data['landscape_generation']

    for i in parser._actions:
        if i.dest == 'dataroot':
            i.default = dataroot
            i.required = False



    #parser.add_argument('--dataroot', default=data_dir)
    print(parser)


    opt, _ = parser.parse_args()
    print(opt)
