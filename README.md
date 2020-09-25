# open_AR_Sandbox
Welcome to the Open_AR_Sandbox repository. If you do not know what this is all about, have a look at this video:

[![The CGRE Sandbox in action](https://img.youtube.com/vi/oE3Atw-YvSA/0.jpg)](https://www.youtube.com/watch?v=oE3Atw-YvSA)

[![What is AR-Sandbox?](https://img.youtube.com/vi/RIvYO1lx6vs/0.jpg)](https://www.youtube.com/watch?v=RIvYO1lx6vs)

![Python 3](https://img.shields.io/badge/Python-3-blue.svg)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

Table of Contents
--------
* [Introduction](README.md#introduction)
* [Features](README.md#features)
* [Requirements](README.md#requirements)
* [Installation](README.md#installation)
    * [Standard packages](README.md#standard-packages)
    * [Kinect Installation](README.md#kinect-installation)
        * [Windows](README.md#for-windows)
            * [KinectV1 - Future](README.md#kinect-v1---future)
            * [kinectV2 - PyKinect2](README.md#kinect-v2---pykinect2)
        * [Linux](README.md#for-linux)
            * [KinectV1 - libkinect](README.md#kinect-v1---libfreenect)
            * [kinectV2 - pylibfreenect](README.md#kinect-v2---pylibfreenect2)
* [Git LFS](README.md#git-lfs)
* [External packages](README.md#external-packages)
    * [Gempy](README.md#gempy)
    * [Devito](README.md#devito)
* [Getting Started](README.md#getting-started)
* [Maintainers](README.md#maintainers)
* [Visit us](README.md#visit-us)
    

Introduction
-----------
Augmented Reality Sandboxes are a great tool for science outreach and teaching due to their intuitive and interaction-enhancing operation. Recently AR Sandboxes are becoming increasingly popular as interactive exhibition pieces, teaching aids and toys.

AR Sandboxes consist of a box of sand that can be freely sculpted by hand. The topography of the sand is constantly scanned with a depth camera and a computed image is projected back onto the sand surface, augmenting the sandbox with digital information.

However, most of these common AR Sandboxes are limited to the visualization of topography with contour lines and colors, as well as water simulations on the digital terrain surface. The potential for AR Sandboxes for geoscience education , and especially for teaching struc- tural geology, remains largely untapped.

For this reason, we have developed open-AR-Sandbox, an augmented reality sandbox designed specifically for the use in geoscience education. In addition to the visualization of topography it can display geologic subsurface information such as the outcropping lithology, creating a dynamic and interactive geological map. The relations of subsurface structures, topography and outcrop can be explored in a playful and comprehensible way.

Features
-------
* compatible with most AR Sandbox builds
* subroutine for calibration and alignment of depth image, sand surface and projection 
* versatile model creation with the powerful GemPy library
* open-source under LGPL v3.0 license
* fully customizable color map, contours and fault line visualization
* We recently added computer vision algorithms to the sandbox that open up a whole new field of possibilities! By placing printed markers into the sandbox, thew user can trigger actions or define points, lines and areas in the sandbox without using the computer

Some of the modules already implemented include:
* [TopoModule](notebooks/tutorials/02_TopoModule/): Normalize the depth image to display a topography map with fully customizable contour lines and variable heights.
* [GradientModule](notebooks/tutorials/05_GradientModule): Takes the gradient information from the depth image and highlight slopes in x and y direction, calculation of laplacian, interactive hill shading, visualization of a vector field, and streamline plot.   
* [LoadSaveTopoModule](notebooks/tutorials/06_LoadSaveTopoModule): Takes the depth image and allows it to be saved as a DEM to reconstruct topographies previously constructed.
* [LandslideSimulation](notebooks/tutorials/07_LandslideSimulation): With precomputed landslides simulations, recreate a topography and trigger a landslide to visualize its flow, direction, and velocity in real-time, or frame by frame.
* [SearchMethodsModule](notebooks/tutorials/03_SearchMethodsModule): Takes the depth image and performs Monte-Carlo simulation algorithms to construct the probability distribution based on the structure of the current DEM in an interactive way. (https://chi-feng.github.io/mcmc-demo/app.html)
* [PrototypingModule](notebooks/tutorials/08_PrototypingModule): Create your own module with the help of this module to link the live threading of the sandbox with your ideas
* [GemPyModule](notebooks/tutorials/04_GempyModule): Use the full advantage of the powerful [GemPy](https://github.com/cgre-aachen/gempy) package to construct geological models and visualize them on the sandbox in real-time
* SeismicModule: module for seismic wave modelling in the sandbox
* LandscapeModule: Landscape generations using machine learning codes extracted from the sandbox
* MarkerDetection: Place virtual boreholes in the model, Define a cross section with multiple markers, Set the start position for simulations (landslides, earthquakes, etc.) check the arucos marker detection for more information (https://docs.opencv.org/trunk/d5/dae/tutorial_aruco_detection.html)



Check the video below for some of the features in action:
[![Open AR Sandbox Features](https://img.youtube.com/vi/t0fyPVMIH4g/0.jpg)](https://www.youtube.com/watch?v=t0fyPVMIH4g)

The open_AR_Sandbox as well as GemPy are under continuous development and including more modules for major outreach. Some of the features we are currently working on include: 

* More Tutorials, examples, Tests and Documentation to help you develop your own modules
* GemPy optimization for (much!) higher framerates
* on-the-fly modification of the geological model (layer dip, thickness fault throw, etc.)
* Integration of more depth sensors (support to all kinect sensors)
* Improve compatibility with Linux and MacOS
* ...


Requirements
--------
You will need: 
* Microsoft Kinect (we tested the first and second generation kinect with a usb adapter, but every kinect compatible with the pykinect drivers will likely work).
* Projector
* A box of Sand

Mount the kinect and projector facing down vertically in the center above of the box. The optimal distance will depend on the size of your sandbox and the optics of the projector, from our experience a distance of 150 cm is well suited for a 80 cm x 100 cm box. 
More details on how to set up the kinect and projector can be found in the `calib_sensor.ipynb` and `calib_projector.ipynb` notebooks.


Installation 
-----
First of all you will need a healthy Python 3 environment. We recommend using [Anaconda](https://www.anaconda.com/distribution/). In addition to some standard Python packages, you will need a specific setup dependent on the Kinect version you are using. In the following we provide detailed installation instructions.\
Now download or clone this repository [open_AR_Sandbox](https://github.com/cgre-aachen/open_AR_Sandbox) from github.

1. First clone the repository:
```
git clone https://github.com/cgre-aachen/open_AR_Sandbox.git
```
2. Create a new anaconda environment
```
conda create -n sandbox-env python=3.7
```
3. Now when you want to use the sandbox and the packages we are about to installl you will have to activate the environment before starting anything
```
conda activate sandbox-env
```
### Standard packages

The standard packages 

```conda install numpy pandas jupyter notebook scipy panel scikit-image matplotlib```

```pip install numpy pandas jupyter notebook scipy panel scikit-image matplotlib```

or simply use our `requirements.txt file`

```pip install -r requirements.txt file```

You can also have a local installation of the sandbox by using the File "setup.py" by doing:

`pip install -e . `

### Kinect Installation 
 
### For Windows

#### Kinect v1 - Future

There is still no support for kinect V1... 

#### Kinect V2 - PyKinect2

(Tested on Windows 10). First, **install the current** [Kinect SDK](https://www.microsoft.com/en-us/download/confirmation.aspx?id=44561) **including drivers**. You can use the software bundle to test the connection to your
 kinect, before you continue.

To make Python and the Kinect SDK communicate, install the related [PyKinect2](https://github.com/Kinect/PyKinect2) wrappers which can be easily installed via:

```pip install pykinect2```

Unfortunately, the configuration of PyKinect2 needs to be adjusted to work on a 64 bit System.  Therefore, edit the
 _Lib/site-packages/pykinect2/PyKinectV2.py_ file, go to line **2216** and comment it:

```python
# assert sizeof(tagSTATSTG) == 72, sizeof(tagSTATSTG)
```

Add the following lines below:

```python
import numpy.distutils.system_info as sysinfo
required_size = 64 + sysinfo.platform_bits / 4
assert sizeof(tagSTATSTG) == required_size, sizeof(tagSTATSTG)
```

### For Linux
!!!Not stable!!!

#### Kinect v1 - libfreenect

To make open_AR_Sandbox talk to the first generation kinect you will need the
[Libfreenect Drivers](https://github.com/OpenKinect/libfreenect) with
[Python Wrappers](https://openkinect.org/wiki/Python_Wrapper). 
The installation is kind of straight forward for Linux and MacOS but 
challenging for Microsoft (in fact: if you pull it off, let us know how you did it!)

#### Kinect v2 - freenect2
or pylibfreenect2 \
For this we are going to use a python interface for the library [libfreenect2](https://github.com/OpenKinect/libfreenect2)
called [freenect2](https://github.com/rjw57/freenect2-python). 
* First we need to install the [freenect2](https://github.com/rjw57/freenect2-python) as described in the installation guide. 
The steps can be summarized as follows (refer to any problems regarding installation in to [link](https://rjw57.github.io/freenect2-python/))
```
git clone https://github.com/OpenKinect/libfreenect2.git
cd libfreenect2
```
```
sudo apt-get install build-essential cmake pkg-config
sudo apt-get install libusb-1.0-0-dev libturbojpeg0-dev libglfw3-dev
```
* With all the dependencies installed now we can make and install 
```
mkdir build && cd build
cmake .. -DENABLE_CXX11=ON -DENABLE_OPENCL=ON -DENABLE_OPENGL=ON -DBUILD_OPENNI2_DRIVER=ON -DCMAKE_INSTALL_PREFIX=$HOME/freenect2 -DCMAKE_VERBOSE_MAKEFILE=ON
make
make install
```
* Set up udev rules for device access: `sudo cp ../platform/linux/udev/90-kinect2.rules /etc/udev/rules.d/, then replug the Kinect.
* Now test if the kinect is correctly installed, run:
```
./bin/Protonect
```
* You should be able to see the kinect image working. If not, check  [libfreenect2](https://github.com/OpenKinect/libfreenect2) 
installation guide for more detailed instructions of installation

* If everything is working until now, we can install the python wrapper. For this first we need to indicate where the `freenect2` folder can be found.
```
export PKG_CONFIG_PATH=$HOME/freenect2/lib/pkgconfig
```
NOTE: if you installed the `freenect2` in other location, specify variables with the corresponding path
* now we can use `pip install` , or any other method described in the [freenect2](https://github.com/rjw57/freenect2-python) 
installation guide. 
```
pip install freenect2
```
IMPORTANT: To this point will work in any python that starts with the terminal. Nevertheless, if we start python from another source, the error 
`ImportError: libfreenect2.so.0.2: cannot open shared object file: No such file or directory` will appear every time we import the package. To fix this problem we will need
to export the variables again or if you want a more permanent solution, open the `.bashrc` file and paste the following at the end of the file:
```
# set PATH to freenect2 to be imported in python
export PKG_CONFIG_PATH=$HOME/freenect2/lib/pkgconfig
```
* With this it will always work for any python open from the terminal. Including jupyter notebooks
* But now if we want to run this package in Pycharm or symilar, we can directly copy the 3 files (`libfreenect2.2.s0...`) from the `freenect2/lib` folder into the 
`lib` folder of your environment. Ej:
 * if you are using an anaconda environment, open the folder:
```
<your_path>/anaconda3/envs/<sandbox-env>/lib
```
* And in this folder paste the previous copied files (3 files!!!). Keep in mind that you need to 
replace the <...> with your specific path.
* If you dont want the manual work then run directly (remember to change the paths according to your needs):
```
sudo cp $HOME/freenect2/lib/libfreenect2{.so,.so.0.2,.so.0.2.0} $HOME/anaconda3/envs/sandbox-env/lib/
```


Git LFS
-------

To clone and use this repository, and specially have the landslides simulations and run the tests, you'll need Git Large File Storage (LFS).

Our [Developer Guide](https://developer.lsst.io/tools/git_lfs.html)
explains how to set up Git LFS for LSST development.

#### Windows
1. Download the windows installer from [here](https://github.com/git-lfs/git-lfs/releases)
2. Run the windows installer    
3. Start a command prompt/or git for windows prompt and run git lfs install`


##### Linux

```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
```



External Packages
---------
### GemPy

To use implicit geological models inside the sandbox, go to [GemPy](https://github.com/cgre-aachen/gempy),
clone or download the repository and follow the Gempy Installation instructions. With gempy installed 
you can follow the tutorial [GempyModule](notebooks/tutorials/04_GempyModule/).
```
pip install gempy
```
### Devito

This package uses the power of [Devito](https://github.com/devitocodes/devito) to run wave proppagation simmulations.
More about this can be found in `notebooks/tutorials/10_SeismicModule/`. Follow the Devito installation instructions. 
* This module so far have only support in linux 
```
git clone https://github.com/devitocodes/devito.git
cd devito
pip install -e .
```

Getting started
-------

So now the necessary software is installed and (hopefully) running and you have set up your Sandbox with a projector and a kinect, it is time to calibrate your sandbox.
The calibration step is necessary to align your sandbox with the kinect as well as with the projected image. 

Navigate to the Folder `/notebooks/tutorials/01_CalibModule` and follow the instruction in the `calib_sensor.ipynb` and `calib_projector.ipynb` notebook. 
If everything goes right you will end up wit a calibration file that you can use for your Sandbox setup.

The next (and most exciting) step is to make the sandbox run and actually showing some geology: the Folder `/notebooks/tutorials/` will guide you through all the available modules and the necessary steps. 


Maintainers
------
* Daniel Escallón Botero
* Simon Virgo
* Miguel de la Varga

Visit us
------
https://www.gempy.org/ar-sandbox

 
