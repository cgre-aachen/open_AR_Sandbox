.. AR_Sandbox documentation master file, created by
   sphinx-quickstart on Tue Apr 14 17:11:54 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Download sample data
====================

You have the option to download some publicly shared files from our
`open_AR_Sandbox <https://github.com/cgre-aachen/open_AR_Sandbox>`_ shared folder. You will need to do this if you want
to run the tests, use the landslides simulations and/or get the trained models for the the use of the Landscape
generation module.

In the terminal type::

   python3 sandbox/utils/download_sample_datasets.py

and follow the instruction on the terminal to download the specific files you need. We use Pooch to help us fetch our
data files and store them locally in your computer to their respective folders. Running this code a second time will not
trigger a download since the file already exists.