.. AR_Sandbox documentation master file, created by
   sphinx-quickstart on Tue Apr 14 17:11:54 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

External packages
=================

GemPy
~~~~~

To use implicit geological models inside the sandbox, go to GemPy, clone or download the repository and follow the
GemPy Installation instructions. With GemPy installed you can follow the tutorial GempyModule::

   pip install gempy

If using windows you will need to install Theano separately as instructed in here::

   conda install mingw libpython m2w64-toolchain
   conda install theano
   pip install theano --force-reinstall

Devito
~~~~~~

This package uses the power of Devito to run wave proppagation simmulations. More about this can be found in
notebooks/tutorials/10_SeismicModule/. Follow the Devito installation instructions. This module so far have only support
in Linux::

   pip install --user git+https://github.com/devitocodes/devito.git

PyGimli
~~~~~~~

This library is a powerful tool for Geophysical inversion and Modelling. Some examples can be found in
notebooks/tutorials/11_Geophysics/. PyGimli can be installed following the installation intructions here. We recomend
creating a new environment where PyGimli is already installed and over that one install the sandbox dependencies::

   conda create -n sandbox-env -c gimli -c conda-forge pygimli=1.1.0

And now go back to installation and follow all over again the instruction but skipping step 2::

   PyTorch

To use the LandscapeGeneration module we need to install PyTorch. This module use the power of CycleGAN to take a
topography from the sandbox, translate this as a DEM and then display it again on the sandbox as a Landscape image.
To install the dependencies for this module do:

- For Windows::

   pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

- For Linux::

   pip install torch torchvision
   git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
   cd pytorch-CycleGAN-and-pix2pix
   pip install -r requirements.txt

Once this is installed, copy the trained model in /notebooks/tutorials/09_LandscapeGeneration/checkpoints folder, and
then follow the notebook. Get in contact with us to provide you with the train model for this module.