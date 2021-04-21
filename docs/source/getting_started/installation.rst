.. AR_Sandbox documentation master file, created by
   sphinx-quickstart on Tue Apr 14 17:11:54 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Installation
============

First of all you will need a healthy `Python 3 <https://www.python.org/>`_ environment. We recommend using
`Anaconda <https://www.anaconda.com/>`_. In addition to some standard `Python 3 <https://www.python.org/>`_ packages,
you will need a specific setup dependent on the Kinect version you are using. In the following we provide detailed
installation instructions.

open_AR_Sandbox package
~~~~~~~~~~~~~~~~~~~~~~~

Download or clone this repository `open_AR_Sandbox <https://github.com/cgre-aachen/open_AR_Sandbox>`_ from GitHub.

First: Clone the repository::

   git clone https://github.com/cgre-aachen/open_AR_Sandbox.git

Second: Create a new anaconda environment::

   conda create -n sandbox-env python

Third: When you want to use the sandbox and the packages we are about to install you will have to activate the
environment before starting anything::

   conda activate sandbox-env


Standard packages
~~~~~~~~~~~~~~~~~

To install all the standard packages please use the requirements.txt file::

   pip install -r requirements.txt

You can also have a local installation of the sandbox by using the File "setup.py" by doing::

   pip install -e


Kinect
~~~~~~

For Windows
^^^^^^^^^^^

- Kinect v1 - Future
There is still no support for Kinect v1!

- Kinect v2 - PyKinect2 (tested on Windows 10)
Install the current Kinect SDK including drivers. You can use the software bundle to test the connection to your kinect,
before you continue.

To make Python and the Kinect SDK communicate, install the related PyKinect2 wrappers which can be easily installed
via::

   pip install pykinect2

Unfortunately, the configuration of PyKinect2 needs to be adjusted to work on a 64 bit System. Therefore, edit the
Lib/site-packages/pykinect2/PyKinectV2.py file, go to line 2216 and comment it::

   # assert sizeof(tagSTATSTG) == 72, sizeof(tagSTATSTG)

Add the following lines below::

   import numpy.distutils.system_info as sysinfo
   required_size = 64 + sysinfo.platform_bits / 4
   assert sizeof(tagSTATSTG) == required_size, sizeof(tagSTATSTG)


For Linux
^^^^^^^^^

- Kinect v1 - libfreenect
To make `open_AR_Sandbox <https://github.com/cgre-aachen/open_AR_Sandbox>`_ talk to the first generation kinect you will
need the Libfreenect Drivers with Python Wrappers. The installation is kind of straight forward for Linux and MacOS but
challenging for Microsoft (in fact: if you pull it off, let us know how you did it!) The steps can be summarized as
follows (refer to any problems regarding installation in to link) To build libfreenect, you'll need

-> libusb >= 1.0.18 (Windows needs >= 1.0.22)
-> CMake >= 3.12.4 (you can visit this page for detailed instructions for the installation)

Once these are installed we can follow the next commands::

   sudo apt-get install git cmake build-essential libusb-1.0-0-dev
   sudo apt-get install freeglut3-dev libxmu-dev libxi-dev
   git clone https://github.com/OpenKinect/libfreenect
   cd libfreenect
   mkdir build
   cd build
   cmake -L .. # -L lists all the project options
   cmake .. -DBUILD_PYTHON3=ON
   make


   cd ../wrappers/python
   python setup.py install
   # now you can see if the installation worked running an example
   python demo_cv2_async.py


- Kinect v2 - freenect2 or pylibfreenect2

For this we are going to use a python interface for the library libfreenect2 called freenect2.

First we need to install the freenect2 as described in the installation guide. The steps can be summarized as follows
(refer to any problems regarding installation in to link)::

   git clone https://github.com/OpenKinect/libfreenect2.git
   cd libfreenect2

   sudo apt-get install build-essential cmake pkg-config
   sudo apt-get install libusb-1.0-0-dev libturbojpeg0-dev libglfw3-dev


With all the dependencies installed now we can make and install::

   mkdir build && cd build
   cmake .. -DENABLE_CXX11=ON -DENABLE_OPENCL=ON -DENABLE_OPENGL=ON -DBUILD_OPENNI2_DRIVER=ON -DCMAKE_INSTALL_PREFIX=$HOME/freenect2 -DCMAKE_VERBOSE_MAKEFILE=ON
   make
   make install


Set up udev rules for device access::

   sudo cp ../platform/linux/udev/90-kinect2.rules /etc/udev/rules.d/

Now unplug and replug the Kinect sensor.

Test if the kinect is correctly installed, by running::

   ./bin/Protonect

You should be able to see the kinect image working. If not, check libfreenect2 installation guide for more detailed
instructions of installation.

If everything is working until now, we can install the python wrapper. For this first we need to indicate where the
freenect2 folder can be found::

   export PKG_CONFIG_PATH=$HOME/freenect2/lib/pkgconfig

NOTE: If you installed the freenect2 in other location, specify variables with the corresponding path

Now we can use pip install, or any other method described in the freenect2 installation guide::

   pip install freenect2

IMPORTANT: To this point will work in any python that starts with the terminal. Nevertheless, if we start python from
another source, the error ImportError: libfreenect2.so.0.2: cannot open shared object file: No such file or directory
will appear every time we import the package. To fix this problem we will need to export the variables again or if you
want a more permanent solution, open the .bashrc file and paste the following at the end of the file::

   # set PATH to freenect2 to be imported in python
   export PKG_CONFIG_PATH=$HOME/freenect2/lib/pkgconfig

With this it will always work for any python open from the terminal. Including jupyter notebooks

But now if we want to run this package in Pycharm or symilar, we can directly copy the 3 files (libfreenect2.2.s0...)
from the freenect2/lib folder into the lib folder of your environment. For instance, if you are using an anaconda
environment, open the folder::

   <your_path>/anaconda3/envs/<sandbox-env>/lib

In this folder paste the previous copied files (3 files!!!). Keep in mind that you need to replace the <...> with your
specific path. If you dont want the manual work then run directly (remember to change the paths according to your
needs)::

   sudo cp $HOME/freenect2/lib/libfreenect2{.so,.so.0.2,.so.0.2.0} $HOME/anaconda3/envs/sandbox-env/lib/

LiDAR L515
~~~~~~~~~~

For Windows
^^^^^^^^^^^
First, go to the latest release page on GitHub and download and execute the file::

   Intel.RealSense.Viewer.exe

Follow the instructions for the installation and update the firmware of your sensor. You should be able to use and see
the depth and RGB image.

For Linux
^^^^^^^^^
Detailed installation steps can be found in the linux installation guide. The steps are as follows:

Register the server's public key::

   sudo apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE

In case the public key still cannot be retrieved, check and specify proxy settings::

   export http_proxy="http://<proxy>:<port>"

and rerun the command. See additional methods in the following link.

Add the server to the list of repositories:

Ubuntu 16 LTS::

   sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo xenial main" -u

Ubuntu 18 LTS::

   sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo bionic main" -u

Ubuntu 20 LTS::

   sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo focal main" -u

Install the libraries::

   sudo apt-get install librealsense2-dkms
   sudo apt-get install librealsense2-utils

Reconnect the Intel RealSense depth camera and run::

   realsense-viewer

to verify the installation.

Running with python
^^^^^^^^^^^^^^^^^^^

After the sensor is installed on your pltaform, the Python wrapper can be easily installed via::

   pip install pyrealsense2

If any problems with the installation reference to Intel RealSense Python Installation