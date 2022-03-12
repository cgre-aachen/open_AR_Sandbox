# Open AR-Sandbox

* Check the complete and up-to-date documentation on [read the docs](https://open-ar-sandbox.readthedocs.io/en/latest/)

Welcome to the# Open AR-Sandbox repository. If you do not know what this is all about, have a look at this video:

[![The CGRE Sandbox in action](https://img.youtube.com/vi/oE3Atw-YvSA/0.jpg)](https://www.youtube.com/watch?v=oE3Atw-YvSA)

[![What is an AR-sandbox?](https://img.youtube.com/vi/RIvYO1lx6vs/0.jpg)](https://www.youtube.com/watch?v=RIvYO1lx6vs)

![Python 3](https://img.shields.io/badge/Python-3-blue.svg)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

Table of Contents
--------
* [Introduction](README.md#introduction)
* [Features](README.md#features)
* [License, use and attributions](README.md#license-use-and-attribution)
* [Requirements](README.md#requirements)
* [Installation](README.md#installation)
    * [Standard packages](README.md#standard-packages)
    * [Download sample data](README.md#download-sample-data)
    * [Kinect Installation](README.md#kinect-installation)
        * [Windows](README.md#for-windows)
            * [KinectV2 - PyKinect2](README.md#kinect-v2---pykinect2)
        * [Linux](README.md#for-linux)
            * [KinectV1 - libkinect](README.md#kinect-v1---libfreenect)
            * [KinectV2 - freenect2](README.md#kinect-v2---freenect2)
   * [LiDAR L515 Installation](README.md#lidar-l515-installation)
      * [Installing in Windows](README.md#installing-in-windows)
      * [Installing in Linux](README.md#installing-in-linux)
      * [Running with Python](README.md#running-with-python)
* [External packages](README.md#external-packages)
    * [Gempy](README.md#gempy)
    * [Devito](README.md#devito)
    * [PyGimli](README.md#pygimli)
    * [PyTorch](README.md#pytorch)
* [Project Development](README.md#project-development)
* [Interested in obtaining a fully operational system?](README.md#obtaining-a-full-system)
    

:warning: **Warning!** It is unfortunate that we have to state this here, but: downloading the software and presenting it somewhere as your 
own work is serious **scientific fraud**! And if you develop content further, then please push these developments
back to this repostory - in the very interest of scientific development (and also a requirement of the license).
For more details, please consult the information below and the license.


Introduction
-----------
Augmented Reality Sandboxes (AR-sandboxes) are a great tool for science outreach and teaching due to their intuitive and interaction-enhancing operation. Recently AR Sandboxes are becoming increasingly popular as interactive exhibition pieces, teaching aids and toys.

AR-sandboxes consist of a box of sand that can be freely sculpted by hand. The topography of the sand is constantly scanned with a depth camera and a computed image is projected back onto the sand surface, augmenting the sandbox with digital information.

However, most of these common AR Sandboxes are limited to the visualization of topography with contour lines and colors, as well as water simulations on the digital terrain surface. The potential for AR Sandboxes for geoscience education , and especially for teaching strutural geology, remains largely untapped.

For this reason, we have developed Open AR-Sandbox, an augmented reality sandbox designed specifically for the use in geoscience education. In addition to the visualization of topography it can display geologic subsurface information such as the outcropping lithology, creating a dynamic and interactive geological map. The relations of subsurface structures, topography and outcrop can be explored in a playful and comprehensible way.

Features
-------
* compatible with most AR-sandbox builds
* subroutine for calibration and alignment of depth image, sand surface and projection 
* versatile model creation with the powerful GemPy library
* open-source under LGPL v3.0 license
* fully customizable color map, contours and fault line visualization
* We recently added computer vision algorithms to the sandbox that open up a whole new field of possibilities! By placing printed markers into the sandbox, thew user can trigger actions or define points, lines and areas in the sandbox without using the computer

Some of the modules already implemented include:
* [MarkerDetection](notebooks/tutorials/00_Calibration): Place virtual boreholes in the model, Define a cross section with multiple markers, Set the start position for simulations (landslides, earthquakes, etc.) check the arucos marker detection for more information (https://docs.opencv.org/trunk/d5/dae/tutorial_aruco_detection.html)
* [TopoModule](notebooks/tutorials/02_TopoModule/): Normalize the depth image to display a topography map with fully customizable contour lines and variable heights.
* [SearchMethodsModule](notebooks/tutorials/03_SearchMethodsModule): Takes the depth image and performs Monte-Carlo simulation algorithms to construct the probability distribution based on the structure of the current DEM in an interactive way. (https://chi-feng.github.io/mcmc-demo/app.html)
* [GemPyModule](notebooks/tutorials/04_GempyModule): Use the full advantage of the powerful [GemPy](https://github.com/cgre-aachen/gempy) package to construct geological models and visualize them on the sandbox in real-time
* [GradientModule](notebooks/tutorials/05_GradientModule): Takes the gradient information from the depth image and highlight slopes in x and y direction, calculation of laplacian, interactive hill shading, visualization of a vector field, and streamline plot.   
* [LoadSaveTopoModule](notebooks/tutorials/06_LoadSaveTopoModule): Takes the depth image and allows it to be saved as a DEM to reconstruct topographies previously constructed.
* [LandslideSimulation](notebooks/tutorials/07_LandslideSimulation): With precomputed landslides simulations, recreate a topography and trigger a landslide to visualize its flow, direction, and velocity in real-time, or frame by frame.
* [PrototypingModule](notebooks/tutorials/08_PrototypingModule): Create your own module with the help of this module to link the live threading of the sandbox with your ideas
* [LandscapeModule](notebooks/tutorials/09_LandscapeGeneration): Landscape generations using machine learning codes powered by [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) 
* [SeismicModule](notebooks/tutorials/10_SeismicModule): module for seismic wave modelling in the sandbox. This uses the power of [Devito](https://github.com/devitocodes/devito)
* [GeoelectricsModule](notebooks/tutorials/11_GeoelectricsModule): module for visualization of geoelectrical fields using aruco markers as electrodes. This use power of [PyGimli](https://www.pygimli.org/)

Check the video below for some of the features in action:
[![Open AR Sandbox Features](https://img.youtube.com/vi/t0fyPVMIH4g/0.jpg)](https://www.youtube.com/watch?v=t0fyPVMIH4g)

The Open AR-Sandbox software as well as GemPy are under continuous development and including more modules for major outreach. Some of the features we are currently working on include: 

* More Tutorials, examples, Tests and Documentation to help you develop your own modules
* GemPy optimization for (much!) higher framerates
* on-the-fly modification of the geological model (layer dip, thickness fault throw, etc.)
* Integration of more depth sensors (support to all kinect sensors)
* Improve compatibility with Linux and MacOS
* ...

License, use and attribution
----------------------------


If you use Open AR-Sandbox in a scientific abstract or publication, please
include appropriate recognition of the original work. For the time being,
please cite our [publication](https://pubs.geoscienceworld.org/gsa/geosphere/article/doi/10.1130/GES02455.1/611689/Open-AR-Sandbox-A-haptic-interface-for-geoscience) in the journal Geosphere:

Florian Wellmann, Simon Virgo, Daniel Escallon, Miguel de la Varga, Alexander Jüstel, Florian M. Wagner, Julia Kowalski, Hu Zhao, Robin Fehling, Qian Chen; Open AR-Sandbox: A haptic interface for geoscience education and outreach. Geosphere 2022; doi: https://doi.org/10.1130/GES02455.1

Directly in BibTeX-format:

Last login: Thu Mar 10 16:07:05 on ttys003
cd Do%                                                                          (base) flow@Florians-Air ~ % cd Downloads 
(base) flow@Florians-Air Downloads % ls -ltr 
total 12614016
-rw-r--r--@  1 flow  staff     2201413 Oct 14  2020 IMG_2778.HEIC
drwxr-xr-x@ 13 flow  staff         416 Nov 24  2020 BlenderGIS-225
-rw-r--r--@  1 flow  staff     1627484 Jul 28  2021 IMG_6742.HEIC
-rw-r--r--@  1 flow  staff     2323774 Jul 29  2021 IMG_6768.HEIC
-rw-r--r--@  1 flow  staff     2531681 Aug 21  2021 IMG_7462.HEIC
-rw-r--r--@  1 flow  staff     2794992 Aug 21  2021 IMG_7464.HEIC
-rw-r--r--@  1 flow  staff     2833877 Aug 21  2021 IMG_7465.HEIC
-rw-r--r--@  1 flow  staff     1867334 Aug 21  2021 IMG_7466.HEIC
-rw-r--r--@  1 flow  staff     2208240 Aug 21  2021 IMG_7467.HEIC
-rw-r--r--@  1 flow  staff      728143 Aug 21  2021 IMG_7468.HEIC
-rw-r--r--@  1 flow  staff      742115 Aug 21  2021 IMG_7469.HEIC
-rw-r--r--@  1 flow  staff      727578 Aug 21  2021 IMG_7470.HEIC
-rw-r--r--@  1 flow  staff     1441489 Aug 21  2021 IMG_7471.HEIC
-rw-r--r--@  1 flow  staff      460722 Aug 21  2021 IMG_7472.HEIC
-rw-r--r--@  1 flow  staff      494645 Aug 21  2021 IMG_7473.HEIC
-rw-r--r--@  1 flow  staff     2756994 Aug 21  2021 IMG_7474.HEIC
-rw-r--r--@  1 flow  staff     1134890 Aug 21  2021 IMG_7475.HEIC
-rw-r--r--@  1 flow  staff      853648 Aug 21  2021 IMG_7476.HEIC
-rw-r--r--@  1 flow  staff     2958846 Aug 21  2021 IMG_7477.HEIC
-rw-r--r--@  1 flow  staff     3260302 Aug 21  2021 IMG_7478.HEIC
-rw-r--r--@  1 flow  staff     3679727 Aug 21  2021 IMG_7479.HEIC
-rw-r--r--@  1 flow  staff     3148930 Aug 21  2021 IMG_7480.HEIC
-rw-r--r--@  1 flow  staff     2104023 Aug 21  2021 IMG_7481.HEIC
-rw-r--r--@  1 flow  staff     3070725 Aug 21  2021 IMG_7482.HEIC
-rw-r--r--@  1 flow  staff     2581082 Aug 21  2021 IMG_7483.HEIC
-rw-r--r--@  1 flow  staff     1206990 Aug 21  2021 IMG_7485.HEIC
-rw-r--r--@  1 flow  staff     2501273 Aug 21  2021 IMG_7486.HEIC
-rw-r--r--@  1 flow  staff     1878645 Aug 21  2021 IMG_7487.HEIC
-rw-r--r--@  1 flow  staff     1851804 Aug 21  2021 IMG_7488.HEIC
-rw-r--r--@  1 flow  staff     1006424 Aug 21  2021 IMG_7489.HEIC
-rw-r--r--@  1 flow  staff      939701 Aug 21  2021 IMG_7490.HEIC
-rw-r--r--@  1 flow  staff      941877 Aug 21  2021 IMG_7491.HEIC
-rw-r--r--@  1 flow  staff     2248061 Aug 21  2021 IMG_7492.HEIC
-rw-r--r--@  1 flow  staff     3256756 Aug 21  2021 IMG_7493.HEIC
-rw-r--r--@  1 flow  staff     3619438 Aug 21  2021 IMG_7494.HEIC
-rw-r--r--@  1 flow  staff     3695819 Aug 21  2021 IMG_7495.HEIC
-rw-r--r--@  1 flow  staff     1955349 Aug 21  2021 90EC44AE-AB83-4643-8590-C63608D81DF7.JPG
-rw-r--r--@  1 flow  staff     2310580 Aug 21  2021 0E1A8AE1-A1C9-424E-8EFB-A284BF373534.JPG
-rw-r--r--@  1 flow  staff     2521117 Aug 21  2021 IMG_7503.HEIC
-rw-r--r--@  1 flow  staff     2555698 Aug 21  2021 IMG_7502.HEIC
-rw-r--r--@  1 flow  staff     2942876 Aug 21  2021 IMG_7504.HEIC
-rw-r--r--@  1 flow  staff     2390769 Aug 21  2021 IMG_7505.HEIC
-rw-r--r--@  1 flow  staff     3337428 Aug 21  2021 IMG_7506.HEIC
-rw-r--r--@  1 flow  staff     3353854 Aug 21  2021 IMG_7507.HEIC
-rw-r--r--@  1 flow  staff     3392136 Aug 21  2021 IMG_7508.HEIC
-rw-r--r--@  1 flow  staff     4626171 Aug 21  2021 IMG_7509.HEIC
-rw-r--r--@  1 flow  staff     2744362 Aug 21  2021 IMG_7510.HEIC
-rw-r--r--@  1 flow  staff     2997335 Aug 21  2021 IMG_7511.HEIC
-rw-r--r--@  1 flow  staff     3990148 Aug 21  2021 IMG_7513.HEIC
-rw-r--r--@  1 flow  staff     4011812 Aug 21  2021 IMG_7512.HEIC
-rw-r--r--@  1 flow  staff     1965233 Aug 21  2021 IMG_7514.HEIC
-rw-r--r--@  1 flow  staff     1979823 Aug 21  2021 IMG_7517.HEIC
-rw-r--r--@  1 flow  staff     1955885 Aug 21  2021 IMG_7516.HEIC
-rw-r--r--@  1 flow  staff     1611730 Aug 21  2021 IMG_7515.HEIC
-rw-r--r--@  1 flow  staff    47230780 Aug 22  2021 1080p.MOV
-rw-r--r--@  1 flow  staff     2521207 Aug 22  2021 IMG_7519.HEIC
-rw-r--r--@  1 flow  staff     3147012 Aug 22  2021 IMG_7520.HEIC
-rw-r--r--@  1 flow  staff     3049934 Aug 22  2021 IMG_7521.HEIC
-rw-r--r--@  1 flow  staff     2478068 Aug 22  2021 IMG_7522.HEIC
-rw-r--r--@  1 flow  staff     2808308 Aug 22  2021 IMG_7523.HEIC
-rw-r--r--@  1 flow  staff     2272202 Aug 22  2021 IMG_7524.HEIC
-rw-r--r--@  1 flow  staff     1933875 Aug 22  2021 IMG_7525.HEIC
-rw-r--r--@  1 flow  staff     1909058 Aug 22  2021 IMG_7526.HEIC
-rw-r--r--@  1 flow  staff     2417415 Aug 22  2021 IMG_7527.HEIC
-rw-r--r--@  1 flow  staff     2172418 Aug 22  2021 IMG_7528.HEIC
-rw-r--r--@  1 flow  staff     1577706 Aug 22  2021 IMG_7529.HEIC
-rw-r--r--@  1 flow  staff      957327 Aug 22  2021 IMG_7530.HEIC
-rw-r--r--@  1 flow  staff      849658 Aug 22  2021 IMG_7531.HEIC
-rw-r--r--@  1 flow  staff     1965641 Aug 22  2021 IMG_7532.HEIC
-rw-r--r--@  1 flow  staff     1895474 Aug 22  2021 IMG_7533.HEIC
-rw-r--r--@  1 flow  staff     1819737 Aug 22  2021 IMG_7534.HEIC
-rw-r--r--@  1 flow  staff      547885 Aug 22  2021 IMG_7535.HEIC
-rw-r--r--@  1 flow  staff      865735 Aug 22  2021 IMG_7536.HEIC
-rw-r--r--@  1 flow  staff     1059911 Aug 22  2021 IMG_7537.HEIC
-rw-r--r--@  1 flow  staff     2366931 Aug 22  2021 IMG_7538.HEIC
-rw-r--r--@  1 flow  staff     1763849 Aug 22  2021 IMG_7539.HEIC
-rw-r--r--@  1 flow  staff     1977762 Aug 22  2021 IMG_7540.HEIC
-rw-r--r--@  1 flow  staff     1570064 Aug 22  2021 IMG_7541.HEIC
-rw-r--r--@  1 flow  staff     3673222 Aug 22  2021 IMG_7542.HEIC
-rw-r--r--@  1 flow  staff     2556829 Aug 22  2021 IMG_7544.HEIC
-rw-r--r--@  1 flow  staff     3026416 Aug 22  2021 IMG_7543.HEIC
-rw-r--r--@  1 flow  staff     3254158 Aug 22  2021 IMG_7545.HEIC
-rw-r--r--@  1 flow  staff     2763951 Aug 22  2021 IMG_7546.HEIC
-rw-r--r--@  1 flow  staff     2660887 Aug 22  2021 IMG_7547.HEIC
-rw-r--r--@  1 flow  staff     1894267 Aug 22  2021 IMG_7548.HEIC
-rw-r--r--@  1 flow  staff     2440140 Aug 22  2021 IMG_7549.HEIC
-rw-r--r--@  1 flow  staff     3934146 Aug 22  2021 IMG_7550.HEIC
-rw-r--r--@  1 flow  staff     3375552 Aug 22  2021 IMG_7551.HEIC
-rw-r--r--@  1 flow  staff     2185802 Aug 22  2021 IMG_7553.HEIC
-rw-r--r--@  1 flow  staff     2107372 Aug 22  2021 IMG_7552.HEIC
-rw-r--r--@  1 flow  staff     2525310 Aug 22  2021 IMG_7554.HEIC
-rw-r--r--@  1 flow  staff     1808974 Aug 22  2021 IMG_7556.HEIC
-rw-r--r--@  1 flow  staff     1813400 Aug 22  2021 IMG_7555.HEIC
-rw-r--r--@  1 flow  staff     3628868 Aug 22  2021 IMG_7558.HEIC
-rw-r--r--@  1 flow  staff     3739892 Aug 22  2021 IMG_7557.HEIC
-rw-r--r--@  1 flow  staff     3621697 Aug 22  2021 IMG_7559.HEIC
-rw-r--r--@  1 flow  staff     1146554 Aug 22  2021 IMG_7560.HEIC
-rw-r--r--@  1 flow  staff     2847991 Aug 22  2021 IMG_7561.HEIC
-rw-r--r--@  1 flow  staff     2812388 Aug 22  2021 IMG_7562.HEIC
-rw-r--r--@  1 flow  staff     1860115 Aug 22  2021 IMG_7564.MOV
-rw-r--r--@  1 flow  staff     2394098 Aug 22  2021 IMG_7563.HEIC
-rw-r--r--@  1 flow  staff     1896411 Aug 22  2021 IMG_7565.HEIC
-rw-r--r--@  1 flow  staff     2168501 Aug 22  2021 IMG_7566.HEIC
-rw-r--r--@  1 flow  staff     2697633 Aug 22  2021 IMG_7567.HEIC
-rw-r--r--@  1 flow  staff     2347308 Aug 22  2021 IMG_7568.HEIC
-rw-r--r--@  1 flow  staff     2594478 Aug 22  2021 IMG_7569.HEIC
-rw-r--r--@  1 flow  staff     2618614 Aug 22  2021 IMG_7570.HEIC
-rw-r--r--@  1 flow  staff     2519283 Aug 22  2021 IMG_7571.HEIC
-rw-r--r--@  1 flow  staff     2574103 Aug 22  2021 IMG_7572.HEIC
-rw-r--r--@  1 flow  staff     2456452 Aug 22  2021 IMG_7573.HEIC
-rw-r--r--@  1 flow  staff     2635947 Aug 22  2021 IMG_5840.JPG
-rw-r--r--@  1 flow  staff     2672491 Aug 22  2021 IMG_5843.JPG
-rw-r--r--@  1 flow  staff     2463250 Aug 22  2021 IMG_5844.JPG
-rw-r--r--@  1 flow  staff     2144335 Aug 22  2021 IMG_5845.JPG
-rw-r--r--@  1 flow  staff     2039261 Aug 22  2021 IMG_5846.JPG
-rw-r--r--@  1 flow  staff     1805177 Aug 22  2021 IMG_5847.JPG
-rw-r--r--@  1 flow  staff     1913889 Aug 22  2021 IMG_5848.JPG
-rw-r--r--@  1 flow  staff     1619158 Aug 22  2021 IMG_7575.HEIC
-rw-r--r--@  1 flow  staff     1610121 Aug 22  2021 IMG_7576.HEIC
-rw-r--r--@  1 flow  staff     2299011 Aug 22  2021 IMG_5850.JPG
-rw-r--r--@  1 flow  staff     3046820 Aug 22  2021 IMG_7578.HEIC
-rw-r--r--@  1 flow  staff     3049555 Aug 22  2021 IMG_7577.HEIC
-rw-r--r--@  1 flow  staff     1357652 Aug 22  2021 IMG_7579.HEIC
-rw-r--r--@  1 flow  staff     1600079 Aug 22  2021 IMG_7580.HEIC
-rw-r--r--@  1 flow  staff     1389666 Aug 22  2021 IMG_7581.HEIC
-rw-r--r--@  1 flow  staff     3135331 Aug 22  2021 IMG_5852.JPG
-rw-r--r--@  1 flow  staff     2861237 Aug 22  2021 IMG_5853.JPG
-rw-r--r--@  1 flow  staff     2157111 Aug 22  2021 IMG_5858.JPG
-rw-r--r--@  1 flow  staff      246629 Aug 22  2021 IMG_5859.JPG
-rw-r--r--@  1 flow  staff     1441434 Aug 22  2021 IMG_5860.JPG
-rw-r--r--@  1 flow  staff     1813766 Aug 22  2021 IMG_5861.JPG
-rw-r--r--@  1 flow  staff     1952748 Aug 22  2021 IMG_5863.JPG
-rw-r--r--@  1 flow  staff     1938426 Aug 22  2021 IMG_5864.JPG
-rw-r--r--@  1 flow  staff     3065880 Aug 22  2021 IMG_5869.JPG
-rw-r--r--@  1 flow  staff     2784107 Aug 22  2021 IMG_5871.JPG
-rw-r--r--@  1 flow  staff     2672181 Aug 22  2021 IMG_5876.JPG
-rw-r--r--@  1 flow  staff     2141576 Aug 22  2021 IMG_5877.JPG
-rw-r--r--@  1 flow  staff     1963433 Aug 22  2021 IMG_5881.JPG
-rw-r--r--@  1 flow  staff     1814269 Aug 22  2021 IMG_5882.JPG
-rw-r--r--@  1 flow  staff     3113712 Aug 22  2021 IMG_5885.JPG
-rw-r--r--@  1 flow  staff     1695675 Aug 22  2021 IMG_7582.HEIC
-rw-r--r--@  1 flow  staff     1619669 Aug 22  2021 IMG_7583.HEIC
-rw-r--r--@  1 flow  staff      629163 Aug 22  2021 IMG_7584.HEIC
-rw-r--r--@  1 flow  staff      351388 Aug 22  2021 IMG_7585.HEIC
-rw-r--r--@  1 flow  staff      360615 Aug 22  2021 IMG_7586.HEIC
-rw-r--r--@  1 flow  staff     1810133 Aug 22  2021 IMG_7587.HEIC
-rw-r--r--@  1 flow  staff     2114918 Aug 22  2021 IMG_7588.HEIC
-rw-r--r--@  1 flow  staff     2176540 Aug 22  2021 IMG_7589.HEIC
-rw-r--r--@  1 flow  staff     1943387 Aug 22  2021 IMG_7590.HEIC
-rw-r--r--@  1 flow  staff     1843587 Aug 22  2021 IMG_7591.HEIC
-rw-r--r--@  1 flow  staff     3590104 Aug 22  2021 IMG_7592.HEIC
-rw-r--r--@  1 flow  staff     1303481 Aug 22  2021 IMG_7593.HEIC
-rw-r--r--@  1 flow  staff     1323364 Aug 22  2021 IMG_7594.HEIC
-rw-r--r--@  1 flow  staff     1317222 Aug 22  2021 IMG_7595.HEIC
-rw-r--r--@  1 flow  staff     1548125 Aug 22  2021 IMG_7596.HEIC
-rw-r--r--@  1 flow  staff     1561555 Aug 22  2021 IMG_7597.HEIC
-rw-r--r--@  1 flow  staff    19454344 Aug 22  2021 IMG_7599.MOV
-rw-r--r--@  1 flow  staff      407407 Aug 22  2021 IMG_7600.HEIC
-rw-r--r--@  1 flow  staff      504725 Aug 22  2021 IMG_7601.HEIC
-rw-r--r--@  1 flow  staff      469858 Aug 22  2021 IMG_7602.HEIC
-rw-r--r--@  1 flow  staff      541668 Aug 22  2021 IMG_7603.HEIC
-rw-r--r--@  1 flow  staff      557525 Aug 22  2021 IMG_7604.HEIC
-rw-r--r--@  1 flow  staff      477543 Aug 22  2021 IMG_7605.HEIC
-rw-r--r--@  1 flow  staff      526421 Aug 22  2021 IMG_7606.HEIC
-rw-r--r--@  1 flow  staff      347576 Aug 22  2021 IMG_7607.HEIC
-rw-r--r--@  1 flow  staff      346796 Aug 22  2021 IMG_7608.HEIC
-rw-r--r--@  1 flow  staff      368434 Aug 22  2021 IMG_7609.HEIC
-rw-r--r--@  1 flow  staff      404055 Aug 22  2021 IMG_7610.HEIC
-rw-r--r--@  1 flow  staff      636954 Aug 22  2021 IMG_7611.HEIC
-rw-r--r--@  1 flow  staff      627852 Aug 22  2021 IMG_7612.HEIC
-rw-r--r--@  1 flow  staff      631526 Aug 22  2021 IMG_7613.HEIC
-rw-r--r--@  1 flow  staff      625304 Aug 22  2021 IMG_7614.HEIC
-rw-r--r--@  1 flow  staff     1007229 Aug 22  2021 IMG_7615.HEIC
-rw-r--r--@  1 flow  staff     1854502 Aug 22  2021 IMG_7616.HEIC
-rw-r--r--@  1 flow  staff     1940906 Aug 22  2021 IMG_7617.HEIC
-rw-r--r--@  1 flow  staff     3125076 Aug 22  2021 IMG_5886.JPG
-rw-r--r--@  1 flow  staff     2206608 Aug 25  2021 IMG_7646.HEIC
-rw-r--r--@  1 flow  staff     2367339 Aug 25  2021 IMG_7647.HEIC
-rw-r--r--@  1 flow  staff     1281363 Aug 25  2021 IMG_7648.HEIC
-rw-r--r--@  1 flow  staff     1615123 Aug 25  2021 IMG_7651.HEIC
-rw-r--r--@  1 flow  staff     1790171 Aug 25  2021 IMG_7649.HEIC
-rw-r--r--@  1 flow  staff     1756046 Aug 25  2021 IMG_7652.HEIC
-rw-r--r--@  1 flow  staff     1649631 Aug 25  2021 IMG_7656.HEIC
-rw-r--r--@  1 flow  staff     1543688 Aug 25  2021 IMG_7654.HEIC
-rw-r--r--@  1 flow  staff     1823836 Aug 25  2021 IMG_7659.HEIC
-rw-r--r--@  1 flow  staff     1796801 Aug 25  2021 IMG_7658.HEIC
-rw-r--r--@  1 flow  staff     1564058 Aug 25  2021 IMG_7657.HEIC
-rw-r--r--@  1 flow  staff     1797045 Aug 25  2021 IMG_7661.HEIC
-rw-r--r--@  1 flow  staff     1792119 Aug 26  2021 IMG_7662.HEIC
-rw-r--r--@  1 flow  staff     2677449 Sep 21 08:42 IMG_8467.HEIC
-rw-r--r--@  1 flow  staff     2493448 Sep 21 14:25 IMG_8479.HEIC
-rw-r--r--@  1 flow  staff     4667306 Sep 22 13:13 IMG_8510.HEIC
-rw-r--r--@  1 flow  staff    28644024 Sep 22 13:13 IMG_8511.MOV
-rw-r--r--@  1 flow  staff     2591122 Sep 23 08:29 IMG_8518.HEIC
-rw-r--r--@  1 flow  staff     4841880 Sep 23 10:10 IMG_8523.HEIC
-rw-r--r--@  1 flow  staff     3403827 Sep 23 11:14 IMG_8524.HEIC
-rw-r--r--@  1 flow  staff     2217389 Sep 24 07:24 IMG_8546.HEIC
-rw-r--r--@  1 flow  staff     1900238 Sep 24 07:34 IMG_8548.HEIC
-rw-r--r--@  1 flow  staff     3187432 Sep 24 18:42 IMG_8558.HEIC
-rw-r--r--@  1 flow  staff     1625325 Sep 25 09:38 IMG_8596.HEIC
-rw-r--r--@  1 flow  staff    12858273 Sep 25 11:41 IMG_8603.MOV
-rw-r--r--@  1 flow  staff    12858273 Sep 25 11:41 IMG_8603 2.MOV
-rw-r--r--@  1 flow  staff     4108250 Sep 25 11:53 IMG_8606.HEIC
-rw-r--r--@  1 flow  staff     1824755 Sep 25 15:48 IMG_8621.HEIC
-rw-r--r--@  1 flow  staff     3417197 Sep 25 16:22 IMG_8630.HEIC
-rw-r--r--@  1 flow  staff     3513319 Sep 25 17:06 IMG_8633.HEIC
-rw-r--r--@  1 flow  staff     3799039 Sep 25 17:55 IMG_8639.HEIC
-rw-r--r--@  1 flow  staff   192037528 Nov  7 08:44 googlechrome.dmg
-rw-r--r--@  1 flow  staff   159182666 Nov  7 17:06 Slack-4.21.1-macOS.dmg
-rw-r--r--@  1 flow  staff      275591 Nov  7 17:12 Invitation_FARMIN_DemoExternalWorkshop_Aachen_25-11-2021.pdf
-rw-r--r--@  1 flow  staff       94505 Nov  8 09:23 Florian.jpg
-rw-r--r--@  1 flow  staff     9822718 Nov  8 09:34 PhDBWieneke.pdf
drwx------@  7 flow  staff         224 Nov  9 09:07 ETH-05 21-2_20211029112158
-rw-r--r--@  1 flow  staff      261315 Nov 15 09:24 Microsoft Word - LoS_Daniel_Pflieger.pdf
-rw-r--r--@  1 flow  staff       13942 Nov 15 14:22 eduroam-OS_X-RAU.mobileconfig
-rw-r--r--@  1 flow  staff   130393688 Nov 15 21:56 Firefox 94.0.1.dmg
-rw-r--r--@  1 flow  staff     8087787 Nov 15 22:08 Grundbuch_1.pdf
-rw-r--r--@  1 flow  staff      669160 Nov 15 22:12 20211115_220730.pdf
-rw-r--r--@  1 flow  staff       37912 Nov 17 09:14 Screenshot 2021-11-17 at 09.12.55.png
drwxr-xr-x@ 17 flow  staff         544 Nov 17 09:50 minerals-1406009-for proof-corrected
-rw-r--r--@  1 flow  staff     6207962 Nov 17 09:51 minerals-1406009-for proof-corrected.zip
-rw-r--r--@  1 flow  staff      506433 Nov 17 22:04 download-2.jpg
-rw-r--r--@  1 flow  staff      691549 Nov 17 22:33 download.jpg
-rw-r--r--@  1 flow  staff           9 Nov 18 08:53 Bekanntmachung_Digitale_Geosysteme.pdf
-rw-r--r--@  1 flow  staff     3457437 Nov 18 22:55 Fienen_v53n3p257.pdf
-rw-r--r--@  1 flow  staff       17519 Nov 19 13:42 Untitled 2.png
-rw-r--r--@  1 flow  staff    43966333 Nov 20 09:37 Präsentation_ABCJ-Mitgliederversammlung_2021_v10.pptx
-rw-r--r--@  1 flow  staff       79194 Nov 23 07:06 FEj81-XWYAQXVSE.jpeg
drwxr-xr-x@ 17 flow  staff         544 Nov 23 08:09 minerals-1406009-for proof
-rw-r--r--@  1 flow  staff      251303 Nov 23 09:51 Microsoft Word - LoS_Anne_Laure_Argentin.docx.pdf
-rw-r--r--@  1 flow  staff     2159735 Nov 23 20:33 hess-23-1015-2019.pdf
-rw-r--r--@  1 flow  staff     1506581 Nov 23 20:39 entropy-20-00601-v2.pdf
-rw-r--r--@  1 flow  staff      263014 Nov 26 15:09 2021_IRTG_MIP_Second_Phase.pdf
-rw-r--r--@  1 flow  staff    13085232 Nov 29 08:07 minerals-10-00967.pdf
-rw-r--r--@  1 flow  staff       12664 Nov 29 08:39 feiertage_nordrhein-westfalen_2022_et.ics
-rw-r--r--@  1 flow  staff        1509 Nov 29 08:41 ferien_nordrhein-westfalen_2022.ics
-rw-r--r--@  1 flow  staff     2679567 Nov 29 13:29 IMG_9213.HEIC
-rw-r--r--@  1 flow  staff    21130047 Nov 29 17:56 Altier_Thesis(Final).pdf
-rw-r--r--@  1 flow  staff      308055 Nov 29 19:49 Microsoft Word - Geoverbund_Letter_of_Support_ SALTIRE_Kiri_Rodgers_v01.docx.pdf
-rw-r--r--@  1 flow  staff       72494 Nov 29 20:06 noun_Deep Learning_2424485.png
-rw-r--r--@  1 flow  staff       17993 Nov 29 20:09 noun_vr_4145514.png
-rw-r--r--@  1 flow  staff      163357 Nov 29 20:19 Untitled.pdf
-rw-r--r--@  1 flow  staff     5753438 Nov 30 06:45 Ragonaetal2006.pdf
-rw-r--r--@  1 flow  staff        7133 Nov 30 08:31 Introduction - Inferential statistics.ipynb
-rw-r--r--@  1 flow  staff     2960579 Nov 30 14:04 IMG_9219.HEIC
-rw-r--r--@  1 flow  staff       84905 Nov 30 17:51 visualization.jpg
-rw-r--r--@  1 flow  staff       39453 Dec  1 06:58 6f92c83b-7879-45e3-b33f-9d15a5506453.pdf
-rw-r--r--@  1 flow  staff       39658 Dec  1 15:41 Untitled.png
-rw-r--r--@  1 flow  staff       28143 Dec  1 16:35 Introduction - Inferential statistics -Lösung.ipynb
-rw-r--r--@  1 flow  staff     1257408 Dec  2 16:59 1-s2.0-S0040195111001788-main.pdf
drwxrwxrwx@ 57 flow  staff        1824 Dec  5 15:14 Heitfeldpreis Fotos
-rw-r--r--@  1 flow  staff       39069 Dec  6 19:56 CoverLetter.pdf
-rw-r--r--@  1 flow  staff      100696 Dec  6 20:03 letter_to_the_editor.pdf
-rw-r--r--@  1 flow  staff    17889040 Dec  6 20:08 BBRBUQ.pdf
-rw-r--r--@  1 flow  staff     1686840 Dec  7 07:55 Wellman_GES02455_v1.pdf
-rw-r--r--@  1 flow  staff         345 Dec  7 08:49 webviz-subsurface-vizualizations-hans-kallekleiv.ics
-rw-r--r--@  1 flow  staff    21894377 Dec  7 10:58 BBRBUQ 2.pdf
-rw-r--r--@  1 flow  staff       64388 Dec  7 15:54 Umsatzdetails_DE84500105175423742140_20211207.pdf
-rw-r--r--@  1 flow  staff       64464 Dec  7 15:54 Umsatzdetails_DE84500105175423742140_20211207-2.pdf
-rw-r--r--@  1 flow  staff       64271 Dec  7 15:54 Umsatzdetails_DE84500105175423742140_20211207-3.pdf
-rw-r--r--@  1 flow  staff    63532346 Dec  7 17:09 Pysubdiv_dec20201.mp4
-rw-r--r--@  1 flow  staff       47799 Dec  9 07:09 Part 1 - Data management_Lösung.ipynb
-rw-r--r--@  1 flow  staff       15848 Dec  9 07:09 Abgabe01_2021.ipynb
-rw-r--r--@  1 flow  staff        8982 Dec  9 07:09 Part 1 - Data management.ipynb
-rw-r--r--@  1 flow  staff       13963 Dec  9 10:56 Introduction - Inferential statistics bsp timit.ipynb
-rw-r--r--@  1 flow  staff       42171 Dec 10 08:10 Anmeldebestaetigung.pdf
-rw-r--r--@  1 flow  staff      665947 Dec 10 16:24 online_71449F40-C486-0F42-79BB0B212E200895.pdf
-rw-r--r--@  1 flow  staff      461075 Dec 11 07:32 045e11d3-15e6-4f32-9be0-6805d6eb5993.JPG
-rw-r--r--@  1 flow  staff      664097 Dec 11 07:32 8e383740-4e57-463d-92d6-5f216c3f97b7.JPG
-rw-r--r--@  1 flow  staff      588264 Dec 11 07:32 8dc68005-e409-4ebb-a640-409210c5abc1.JPG
-rw-r--r--@  1 flow  staff      102898 Dec 11 07:32 e10560db-2bc9-4f29-8c16-95648e8f5a79.JPG
-rw-r--r--@  1 flow  staff      119358 Dec 11 07:32 d39de834-0edc-49d6-b1b7-06dab41df3a7.JPG
-rw-r--r--@  1 flow  staff       81621 Dec 11 07:32 ad574c26-7e0b-4c92-9758-a4216fc92730.JPG
-rw-r--r--@  1 flow  staff      146937 Dec 11 07:32 68e26029-ff51-4c14-a936-ead3e3426be9.JPG
-rw-r--r--@  1 flow  staff      264656 Dec 11 07:32 67495be3-fc0a-4a9a-a1df-90261e812610.JPG
-rw-r--r--   1 flow  staff     1381613 Dec 11 07:54 IMG_7484.HEIC
-rw-r--r--   1 flow  staff     1798005 Dec 11 07:59 IMG_7598.HEIC
-rw-r--r--   1 flow  staff     1553219 Dec 11 07:59 IMG_7574.HEIC
drwxr-xr-x@  6 flow  staff         192 Dec 11 16:59 Steiger remix.band
drwxr-xr-x@  6 flow  staff         192 Dec 11 17:05 Steiger remix-2.band
-rw-r--r--@  1 flow  staff      759000 Dec 11 21:25 HiWi Job.pptx
drwxr-xr-x@  6 flow  staff         192 Dec 12 08:32 Steiger remix with vocals.band
-rw-r--r--@  1 flow  staff      295947 Dec 12 09:55 scan_202111181433_72545654090.pdf
-rw-r--r--@  1 flow  staff     1358911 Dec 12 18:54 Steiger short - 12.12.21, 18.54.mp3
drwxr-xr-x@  6 flow  staff         192 Dec 12 19:07 Steiger short.band
-rw-r--r--@  1 flow  staff     7860845 Dec 12 19:37 KP_Gesamt.pdf
-rw-r--r--@  1 flow  staff      756541 Dec 12 20:24 Schadenanzeige Fahrrad - Fahrradvollkasko Privat.pdf
-rw-r--r--@  1 flow  staff      769368 Dec 12 20:27 Schadenanzeige Fahrrad - Fahrradvollkasko Privat 2.pdf
-rw-r--r--@  1 flow  staff      295947 Dec 12 20:34 Strafanzeige_Fahrraddiebstahl_Wellmann
-rw-r--r--@  1 flow  staff      139522 Dec 13 06:30 61500_ger-DE.pdf
-rw-r--r--@  1 flow  staff       11959 Dec 14 13:45 Kurzanleitung SPSS - Datenaufbereitung.docx
-rw-r--r--@  1 flow  staff    43944179 Dec 14 16:00 BlueJeansMeetingInstaller(x86_64).dmg
-rw-r--r--@  1 flow  staff       18438 Dec 14 18:49 c6b5bu0ol3ql9n19nuc0.xlsx
-rw-r--r--@  1 flow  staff     3364389 Dec 15 08:37 A_Novel_and_Open-Source_Illumination_Correction_fo.pdf
-rw-r--r--@  1 flow  staff     2046746 Dec 15 12:25 IMG_9511.HEIC
-rw-r--r--@  1 flow  staff       42913 Dec 16 08:39 Part 2 - Basic statistics_loesung.ipynb
-rw-r--r--@  1 flow  staff        8635 Dec 16 08:39 Part 2 - Basic statistics.ipynb
drwxrwxrwx@  3 flow  staff          96 Dec 16 08:47 Arbeit am Datensatz
-rw-r--r--@  1 flow  staff     1065187 Dec 16 12:55 Template_Reports_Docs_Liang copy.pdf
-rw-r--r--@  1 flow  staff      157424 Dec 17 08:48 Frangos_AnGeom.pdf
-rw-r--r--@  1 flow  staff       13892 Dec 17 16:24 Unknown
-rw-r--r--@  1 flow  staff       24024 Dec 17 17:53 c6b5bu0ol3ql9n19nuc0 (1).xlsx
-rw-r--r--@  1 flow  staff        1508 Dec 19 15:40 HappyBirthday_1.midi
-rw-r--r--   1 flow  staff      490378 Dec 19 19:08 BlenderGIS-225.zip
-rw-r--r--@  1 flow  staff      174088 Dec 19 20:19 FG8PtyQXMAMEalb.jpeg
-rw-r--r--@  1 flow  staff   316280262 Dec 20 09:11 Steiger 2021_11.mp4
-rw-r--r--@  1 flow  staff  1145235968 Dec 20 09:20 Heitfeldpreis Fotos.tar
-rw-r--r--   1 flow  staff     4059937 Dec 20 09:33 IMG_8467 copy.jpg
-rw-r--r--   1 flow  staff     3944959 Dec 20 09:33 IMG_8479 copy.jpg
-rw-r--r--   1 flow  staff     7254532 Dec 20 09:33 IMG_8510 copy.jpg
-rw-r--r--   1 flow  staff     4482777 Dec 20 09:33 IMG_8518 copy.jpg
-rw-r--r--   1 flow  staff     7593817 Dec 20 09:33 IMG_8523 copy.jpg
-rw-r--r--   1 flow  staff     5599659 Dec 20 09:33 IMG_8524 copy.jpg
-rw-r--r--   1 flow  staff     3596796 Dec 20 09:33 IMG_8548 copy.jpg
-rw-r--r--   1 flow  staff     3506037 Dec 20 09:33 IMG_8546 copy.jpg
-rw-r--r--   1 flow  staff     2943026 Dec 20 09:33 IMG_8596 copy.jpg
-rw-r--r--   1 flow  staff     3789958 Dec 20 09:33 IMG_8558 copy.jpg
-rw-r--r--   1 flow  staff     6029246 Dec 20 09:33 IMG_8639 copy.jpg
-rw-r--r--   1 flow  staff     5603813 Dec 20 09:33 IMG_8633 copy.jpg
-rw-r--r--   1 flow  staff     3037893 Dec 20 09:33 IMG_8621 copy.jpg
-rw-r--r--   1 flow  staff     5232812 Dec 20 09:33 IMG_8630 copy.jpg
-rw-r--r--   1 flow  staff     6462138 Dec 20 09:33 IMG_8606 copy.jpg
-rw-r--r--   1 flow  staff       76537 Dec 20 09:35 IMG_8467 copy-1.jpg
-rw-r--r--   1 flow  staff       91128 Dec 20 09:35 IMG_8479 copy-1.jpg
-rw-r--r--   1 flow  staff      132381 Dec 20 09:35 IMG_8510 copy-1.jpg
-rw-r--r--   1 flow  staff      107768 Dec 20 09:35 IMG_8518 copy-1.jpg
-rw-r--r--   1 flow  staff      147814 Dec 20 09:35 IMG_8523 copy-1.jpg
-rw-r--r--   1 flow  staff      125370 Dec 20 09:35 IMG_8524 copy-1.jpg
-rw-r--r--   1 flow  staff       87940 Dec 20 09:35 IMG_8548 copy-1.jpg
-rw-r--r--   1 flow  staff       68329 Dec 20 09:35 IMG_8546 copy-1.jpg
-rw-r--r--   1 flow  staff      111582 Dec 20 09:35 IMG_8596 copy-1.jpg
-rw-r--r--   1 flow  staff       67641 Dec 20 09:35 IMG_8558 copy-1.jpg
-rw-r--r--   1 flow  staff      115818 Dec 20 09:35 IMG_8639 copy-1.jpg
-rw-r--r--@  1 flow  staff      102015 Dec 20 09:35 IMG_8633 copy-1.jpg
-rw-r--r--@  1 flow  staff       77718 Dec 20 09:35 IMG_8621 copy-1.jpg
-rw-r--r--   1 flow  staff       90181 Dec 20 09:35 IMG_8630 copy-1.jpg
-rw-r--r--   1 flow  staff      139352 Dec 20 09:35 IMG_8606 copy-1.jpg
-rw-r--r--   1 flow  staff     3006002 Dec 20 09:37 IMG_9511 copy.jpg
-rw-r--r--@  1 flow  staff       64631 Dec 21 11:47 Umsatzdetails_DE84500105175423742140_20211221.pdf
-rw-r--r--@  1 flow  staff     3493376 Dec 21 12:17 HyChemDaten_bearbeitet_2021_22.xls
-rw-r--r--@  1 flow  staff       37573 Dec 21 12:17 SPSS_Daten_2021_22.sav
-rw-r--r--@  1 flow  staff      167930 Dec 21 12:18 SPSS_Ausgabe_2021_22.spv
-rw-r--r--@  1 flow  staff      262146 Dec 21 14:55 Unterweisung_nach_Anlage_1_zur_Coronateststrukturverordnung.pdf
-rw-r--r--@  1 flow  staff      839040 Dec 21 14:57 gv29-1anlage3.pdf
-rw-r--r--@  1 flow  staff       52563 Dec 21 16:41 image.png
-rw-r--r--@  1 flow  staff   317055978 Dec 21 16:51 Steiger 2021_1080p.mp4
drwx------@  6 flow  staff         192 Dec 22 09:12 Special accomodation request RWTH
-rw-r--r--@  1 flow  staff   119577463 Dec 22 15:22 Steiger 2021_twitter resolution.mp4
-rw-r--r--@  1 flow  staff      201810 Dec 24 09:16 render_cycles_nodes_types_shaders_principled_example-1a.jpg
-rw-r--r--@  1 flow  staff      665951 Dec 24 15:45 online_71449F40-C486-0F42-79BB0B212E200895-2.pdf
-rw-r--r--@  1 flow  staff       21807 Dec 24 16:08 FW Sperrung Eingang Parkplatzseite_1321021281.eml
-rw-r--r--@  1 flow  staff     1761374 Dec 26 09:27 IMG_6594.HEIC
drwxr-xr-x@  5 flow  staff         160 Dec 29 07:57 additional_material
-rw-r--r--@  1 flow  staff      617201 Jan  2 16:20 IMG_0085.jpg
-rw-r--r--@  1 flow  staff      617201 Jan  2 16:20 IMG_0085 2.jpg
-rw-r--r--@  1 flow  staff     3599168 Jan  3 09:38 4779-705122.pdf
-rw-r--r--@  1 flow  staff     1178810 Jan  3 18:55 IMG_7080.JPG
-rw-r--r--@  1 flow  staff     1368737 Jan  3 18:56 IMG_7081.JPG
-rw-r--r--@  1 flow  staff       22813 Jan  4 17:06 abstract_EGU_DanielEscallon.odt
-rw-r--r--@  1 flow  staff           0 Jan  5 09:44 PropagateLogout
-rw-r--r--@  1 flow  staff         967 Jan  6 10:30 rootcert.crt
-rw-r--r--@  1 flow  staff      161313 Jan  7 11:04 paper_ogr_prob_ml_geo.pdf
-rw-r--r--@  1 flow  staff    25073906 Jan  8 07:17 530.Michael R. Matthews.pdf
-rw-r--r--@  1 flow  staff      525035 Jan  8 08:06 463-1.pdf
-rw-r--r--@  1 flow  staff      413683 Jan  9 12:34 MB_115_d.pdf
-rw-r--r--@  1 flow  staff       93466 Jan  9 13:30 Erkl__rung_zum_Verlust_des_Versicherungsscheins.pdf
-rw-r--r--@  1 flow  staff     2521080 Jan 10 08:03 era_2018_journal_list.xlsx
-rw-r--r--@  1 flow  staff     1437294 Jan 10 11:09 Ba Thesis V1.pdf
-rw-r--r--@  1 flow  staff      124245 Jan 10 11:16 04-IAMG2022_LoopStructural_LachlanGrose.pdf
-rw-r--r--@  1 flow  staff      434163 Jan 10 11:17 05-IAMG2022_SKUA_Training_Proposal.pdf
-rw-r--r--@  1 flow  staff     1159477 Jan 10 12:23 IMG_0154.HEIC
-rw-r--r--@  1 flow  staff    22797882 Jan 10 20:09 IMG_7631.jpg
-rw-r--r--@  1 flow  staff    15586189 Jan 10 20:10 IMG_7622.jpg
-rw-r--r--@  1 flow  staff    22441812 Jan 10 20:10 IMG_7594.jpg
-rw-r--r--@  1 flow  staff    17518280 Jan 10 20:10 IMG_7590.jpg
-rw-r--r--@  1 flow  staff     6898411 Jan 10 21:40 Reviews of Geophysics - 2021 - Yu - Deep Learning for Geophysics  Current and Future Trends.pdf
-rw-r--r--@  1 flow  staff      970449 Jan 11 17:49 Toxic Floods _CRC_Concept_Part 2_projects_review.docx
-rw-r--r--@  1 flow  staff      247821 Jan 11 18:17 ticket.pdf
-rw-r--r--@  1 flow  staff      483951 Jan 13 08:37 Cleaned.csv
-rw-r--r--@  1 flow  staff       29589 Jan 13 11:49 Form+Letter+of+Recommendation_New+Version_R.docx
-rw-r--r--@  1 flow  staff       85038 Jan 13 11:52 23834-239-cv.pdf
-rw-r--r--@  1 flow  staff       22194 Jan 13 11:52 23834-485-list-relevant-courses.pdf
-rw-r--r--@  1 flow  staff       60933 Jan 13 11:52 23834-413-motivational-letter-chile.pdf
-rw-r--r--@  1 flow  staff     2059526 Jan 13 11:55 23834-241-transcript-of-records.pdf
-rw-r--r--@  1 flow  staff       58627 Jan 13 12:02 23834-415-motivational-letter-nz.pdf
-rw-r--r--@  1 flow  staff       58844 Jan 13 12:02 23834-417-motivational-letter-vietnam.pdf
-rw-r--r--@  1 flow  staff       34934 Jan 13 12:16 24691-407-daad-moritz-strueve.pdf
-rw-r--r--@  1 flow  staff       92023 Jan 13 15:19 Overview_Modelling-1[2].pdf
-rw-r--r--@  1 flow  staff      550299 Jan 13 15:53 Revision S04.docx
-rw-r--r--@  1 flow  staff     2831921 Jan 14 06:44 JGR Solid Earth - 2021 - Miltenberger - Probabilistic Evaluation of Geoscientific Hypotheses With Geophysical Data .pdf
-rw-r--r--@  1 flow  staff      397026 Jan 15 14:34 UnityDownloadAssistant-2021.2.8f1.dmg
-rw-r--r--@  1 flow  staff   130455086 Jan 15 21:04 UnityHubSetup.dmg
-rw-r--r--@  1 flow  staff     1054615 Jan 17 09:02 dell-24-monitor-p2422h-datasheet.pdf
-rw-r--r--@  1 flow  staff     5055848 Jan 17 10:51 Anleitung Digitale Signatur.20200826.signiert.pdf
-rw-r--r--@  1 flow  staff      115651 Jan 18 08:10 OER-Doodle.jpg
-rw-r--r--@  1 flow  staff     1214364 Jan 18 08:10 BL Slide Fachgruppenrat - Januar.pptx
-rw-r--r--@  1 flow  staff       64292 Jan 18 10:11 Umsatzdetails_DE84500105175423742140_20220118.pdf
-rw-r--r--@  1 flow  staff     8951451 Jan 18 16:00 Webex.pkg
-rw-r--r--@  1 flow  staff    52474903 Jan 18 20:24 minerals_eartharxiv_preprint.pdf
-rw-r--r--@  1 flow  staff      104870 Jan 19 19:26 Template_Reports_Docs_Santoso.pdf
-rw-r--r--@  1 flow  staff       15426 Jan 20 09:27 Template_Reports_Docs_Name.tex
-rw-r--r--@  1 flow  staff     5940608 Jan 20 17:59 Kirkwood2022_Article_BayesianDeepLearningForSpatial.pdf
-rw-r--r--@  1 flow  staff      470128 Jan 20 18:00 Ortiz-Silva2022_ReferenceWorkEntry_Entropy.pdf
-rw-r--r--@  1 flow  staff      411668 Jan 20 20:01 Meeting Mohammad_Densie_Florian.pdf
-rw-r--r--@  1 flow  staff       15565 Jan 20 20:04 Meeting Mohammad_Densie_Florian.docx
-rw-r--r--@  1 flow  staff      201615 Jan 21 12:23 LBEG_Stellenausschreibung_L 71_21.pdf
-rw-r--r--@  1 flow  staff      502660 Jan 21 14:43 998ddc9e-2c9e-423e-9939-3cf98df7da0f.JPG
-rw-r--r--@  1 flow  staff    32321391 Jan 21 16:59 Florian_Konrad_Dissertation_hq_online.pdf
-rw-r--r--@  1 flow  staff     4427864 Jan 21 17:04 Anticline_0_out.e
-rw-r--r--@  1 flow  staff     7108382 Jan 21 17:04 anticline_test_0.msh
-rw-r--r--@  1 flow  staff     1739871 Jan 21 17:18 Ba Thesis V2.2.pdf
-rw-r--r--@  1 flow  staff      841729 Jan 22 16:59 Screenshot 2022-01-22 at 16.50.20.png
-rw-r--r--@  1 flow  staff     4919861 Jan 22 17:19 320651_1_rebuttal_8886344_r3qnq9_convrt.pdf
-rw-r--r--@  1 flow  staff    10420448 Jan 22 17:20 320651_1_merged_1639046432.pdf
-rw-r--r--@  1 flow  staff     3293566 Jan 22 17:33 320651_1_related_ms_8886346_r3sfzs.pdf
-rw-r--r--@  1 flow  staff         317 Jan 24 10:18 scholar.enw.ris
-rw-r--r--@  1 flow  staff      694431 Jan 24 12:16 Scan20220124120324.pdf
-rw-r--r--@  1 flow  staff     3514804 Jan 24 15:31 Santoso_Ryan_Presentation_revised1.pptx
-rw-r--r--@  1 flow  staff      817751 Jan 24 20:15 Scan20220124120324-signed.pdf
-rw-r--r--@  1 flow  staff     3989644 Jan 25 07:45 Santoso_Ryan_Presentation_revised2.pptx
-rw-r--r--@  1 flow  staff    93102112 Jan 25 08:13 EndNoteX9Installer.dmg
-rw-r--r--@  1 flow  staff    93063569 Jan 25 09:13 EndNote20SiteInstaller.dmg
-rw-r--r--@  1 flow  staff         197 Jan 25 09:19 scholar.enw-2.ris
-rw-r--r--@  1 flow  staff      752405 Jan 25 09:20 nature14541.pdf
-rw-r--r--@  1 flow  staff         268 Jan 25 09:21 scholar.enw-3.ris
-rw-r--r--@  1 flow  staff         281 Jan 26 08:18 scholar.enw-4.ris
-rw-r--r--@  1 flow  staff     1565773 Jan 26 08:19 10107939.pdf
-rw-r--r--@  1 flow  staff   560672538 Jan 26 15:08 CEPAX-LAB v.7 final.MP4
-rw-r--r--@  1 flow  staff         155 Jan 26 21:35 scholar.enw-5.ris
-rw-r--r--@  1 flow  staff      263509 Jan 27 05:05 image (1).png
-rw-r--r--@  1 flow  staff      132717 Jan 27 07:29 result.pdf
-rw-r--r--@  1 flow  staff     1089989 Jan 27 08:10 rsta.2012.0222.pdf
-rw-r--r--@  1 flow  staff      361933 Jan 27 10:35 1EB3782B-AB4A-46AF-BCF6-12E93DADAE19_1_105_c.jpeg
-rw-r--r--@  1 flow  staff      334960 Jan 27 10:41 scalar_field.png
-rw-r--r--@  1 flow  staff      132972 Jan 27 10:41 Lith.png
-rw-r--r--@  1 flow  staff     2978789 Jan 27 14:59 Wellmann_et_al_2011__AAPG_Hedberg.pdf
-rw-r--r--@  1 flow  staff     1083098 Jan 27 15:00 Estimates_of_sustainable_pumping_in_Hot_20160129-15961-o0doe7-with-cover-page-v2.pdf
-rw-r--r--@  1 flow  staff         135 Jan 27 20:06 scholar.enw-6.ris
-rw-r--r--@  1 flow  staff      738603 Jan 27 22:09 de-roos21a.pdf
-rw-r--r--@  1 flow  staff      161424 Jan 28 06:56 image (2).png
-rw-r--r--@  1 flow  staff       33533 Jan 28 10:04 Prior_3D.png
-rw-r--r--@  1 flow  staff       53114 Jan 28 10:04 HMC_3D.png
-rw-r--r--@  1 flow  staff       48557 Jan 28 10:04 True_3D.png
-rw-r--r--@  1 flow  staff      150506 Jan 29 15:53 Vorlesungseziten_Nordrhein-Westfalen_2016.pdf
-rw-r--r--@  1 flow  staff     1894837 Jan 30 16:48 treece_tr333.pdf
-rw-r--r--@  1 flow  staff      106845 Jan 30 16:58 gmd-2021-187-author_response-version2.pdf
-rw-r--r--@  1 flow  staff     5382031 Jan 30 17:08 gmd-2021-187-manuscript-version4.pdf
-rw-r--r--@  1 flow  staff      472154 Jan 30 17:22 Wellmann2022_ReferenceWorkEntry_TopologyInGeosciences.pdf
-rw-r--r--@  1 flow  staff      934661 Jan 30 17:34 DG-RR_plus_final_proposal_as_submitted_v20210326.pdf
-rw-r--r--@  1 flow  staff     3971718 Jan 31 09:20 49b84d3c-ca76-4668-8d1d-4b24072a5f38.pdf
-rw-r--r--@  1 flow  staff    38756250 Jan 31 11:48 BiooekonomieREVIER_DG-RR_plus_kickoff_meeting_complete_v13.pdf
-rw-r--r--@  1 flow  staff      738603 Jan 31 13:50 de-roos21a-2.pdf
-rw-r--r--@  1 flow  staff       11902 Jan 31 14:07 SupervisorList.xlsx
-rw-r--r--@  1 flow  staff     1605921 Feb  1 06:06 Wellman_GES02455_v2GH.pdf
-rw-r--r--@  1 flow  staff    11930656 Feb  1 10:34 SkypeMeetingsApp.dmg
-rw-r--r--@  1 flow  staff    11930656 Feb  1 10:35 SkypeMeetingsApp-1.dmg
-rw-r--r--@  1 flow  staff      487301 Feb  1 16:40 image (3).png
-rw-r--r--@  1 flow  staff      684113 Feb  1 16:48 book.pdf
-rw-r--r--@  1 flow  staff     3957064 Feb  1 16:52 deliverable_BGR_01.pdf
-rw-r--r--@  1 flow  staff      118033 Feb  1 18:10 document.pdf
-rw-r--r--@  1 flow  staff    60226257 Feb  2 10:29 Mineral_revision.pdf
-rw-r--r--@  1 flow  staff      121485 Feb  2 10:35 graphical_abstract.pdf
-rw-r--r--@  1 flow  staff     2691458 Feb  2 10:59 ENGEO-D-22-00160_reviewer.pdf
-rw-r--r--@  1 flow  staff   126297745 Feb  2 11:28 CAGEO-D-21-00531_R1.pdf
-rw-r--r--@  1 flow  staff       38130 Feb  2 11:32 supplementary.pdf
-rw-r--r--@  1 flow  staff     1623879 Feb  2 21:53 RWE.pdf
-rw-r--r--@  1 flow  staff   150949351 Feb  3 08:25 Masterarbeit_Alexander_Juestel_Geomodeling_Weisweiler.pdf
-rw-r--r--@  1 flow  staff    18794408 Feb  3 08:34 Welcome_presentation2021.pptx
-rw-r--r--@  1 flow  staff    18794433 Feb  3 08:35 Welcome_farewell2021.pptx
-rw-r--r--@  1 flow  staff     5501356 Feb  3 08:38 Final Get-together - January 2020.pptx
-rw-r--r--@  1 flow  staff     6238514 Feb  3 17:02 HadCRUT5_accepted.pdf
-rw-r--r--@  1 flow  staff      794116 Feb  4 08:02 2022-01-31 GAB_Newsletter Dec - Jan.pdf
-rw-r--r--@  1 flow  staff      268646 Feb  4 09:28 GSA_License_to_Publish_2021.pdf
-rw-r--r--@  1 flow  staff      207649 Feb  4 09:29 GSA_License_to_Publish_2021_signed.pdf
-rw-r--r--@  1 flow  staff     2314307 Feb  5 09:40 EASYGO_10_Mathur_Bakul.pdf
-rw-r--r--@  1 flow  staff      321139 Feb  7 08:06 PhD_position_IRTG_P11_DD_next_phase.pdf
-rw-r--r--@  1 flow  staff      582899 Feb  7 08:52 Attachment 1 EERA Geothermal Annual report 2019 for websites .pdf
-rw-r--r--@  1 flow  staff      235787 Feb  7 08:56 Einstellungsantrag 2022 Niederau.pdf
-rw-r--r--@  1 flow  staff      321177 Feb  7 09:10 Einstellungsantrag 2022 Niederau signed.pdf
-rw-r--r--@  1 flow  staff    29675593 Feb  7 13:25 Radwa AlMoqaddam Field Notes-426621-Allgäu Field Trip-compressed.pdf
-rw-r--r--@  1 flow  staff      189853 Feb  7 16:55 Stufenzuordnung_Niederau.pdf
-rw-r--r--@  1 flow  staff      172982 Feb  7 16:56 Drittmittelanzeige_GeoBlocks korrigiert.pdf
-rw-r--r--@  1 flow  staff      296256 Feb  7 16:59 Stufenzuordnung_Niederau_signed.pdf
-rw-r--r--@  1 flow  staff      358977 Feb  7 17:02 Drittmittelanzeige_GeoBlocks korrigiert_signed.pdf
-rw-r--r--@  1 flow  staff     1407452 Feb  7 17:02 Einstellungsantrag Robin Fehling[SHK].pdf
-rw-r--r--@  1 flow  staff     1461758 Feb  7 17:03 Einstellungsantrag Robin Fehling[SHK]_singned.pdf
-rw-r--r--@  1 flow  staff       16321 Feb  7 21:51 NurGIS-VR Antragstexte.docx
-rw-r--r--@  1 flow  staff     1405821 Feb  8 17:51 IMG_0410.HEIC
-rw-r--r--@  1 flow  staff     2025393 Feb  8 17:51 IMG_0411.HEIC
-rw-r--r--@  1 flow  staff     1699253 Feb  8 23:38 IMG_0414.HEIC
-rw-r--r--@  1 flow  staff     1867356 Feb  9 13:14 IMG_0425.HEIC
-rw-r--r--@  1 flow  staff       51450 Feb 11 15:08 KI-Serviceantrag_LOl-Geoverbund-ABCJ_v01.docx
-rw-r--r--@  1 flow  staff      203950 Feb 11 15:09 KI-Serviceantrag_LOl-Geoverbund-ABCJ_v01.pdf
-rw-r--r--@  1 flow  staff      305822 Feb 11 15:11 KI-Serviceantrag_LOl-Geoverbund-ABCJ_v01_signed.pdf
-rw-r--r--@  1 flow  staff       12007 Feb 12 09:50 7b807661b5ff49550eb077bdc3506a45.jpg
-rw-r--r--@  1 flow  staff       62833 Feb 12 09:51 Adorable-Hopping-Bunny-Coloring-Pages.jpg
-rw-r--r--@  1 flow  staff       71779 Feb 12 09:52 Raskrasil.com-dolphins-102.jpg
-rw-r--r--   1 flow  staff       83155 Feb 12 10:25 IMG_0425 copy.jpg
-rw-r--r--   1 flow  staff       72174 Feb 12 10:25 IMG_0414 copy.jpg
-rw-r--r--   1 flow  staff       73386 Feb 12 10:25 IMG_0411 copy.jpg
-rw-r--r--   1 flow  staff       57158 Feb 12 10:25 IMG_0410 copy.jpg
-rw-r--r--@  1 flow  staff      217995 Feb 14 10:07 FARMIN deliverable D2.1.1.pdf
-rw-r--r--@  1 flow  staff      284492 Feb 14 10:08 EIT deliverable template.docx
-rw-r--r--@  1 flow  staff     1080210 Feb 14 18:29 Saldenbestätigung Handkasse.pdf
-rw-r--r--@  1 flow  staff     1203533 Feb 14 18:30 Saldenbestätigung Handkasse FW.pdf
-rw-r--r--@  1 flow  staff      966562 Feb 15 09:00 70_years_of_machine_learning_in_geoscience_in_revi.pdf
-rw-r--r--@  1 flow  staff     3768915 Feb 15 09:02 main.pdf
-rw-r--r--@  1 flow  staff     1105736 Feb 15 09:16 WB Mosaku.pdf
-rw-r--r--@  1 flow  staff      667740 Feb 15 09:17 aufloesung_pflieger[1].pdf
-rw-r--r--@  1 flow  staff     1422563 Feb 15 09:17 WB Moritz Strüve III.pdf
-rw-r--r--@  1 flow  staff     1802051 Feb 15 09:38 ges02455.pdf
-rw-r--r--@  1 flow  staff     1465965 Feb 15 09:52 WB Moritz Strüve III signed.pdf
-rw-r--r--@  1 flow  staff      783620 Feb 15 09:53 aufloesung_pflieger[1] signed.pdf
-rw-r--r--@  1 flow  staff     1228987 Feb 15 09:54 WB Mosaku signed.pdf
-rw-r--r--@  1 flow  staff     2443071 Feb 15 12:44 IMG_0479.HEIC
-rw-r--r--@  1 flow  staff       89029 Feb 15 17:49 KICN02-10 - DoH - Students & Industry - Knowledge Triangle Integration.docx
-rw-r--r--@  1 flow  staff      142666 Feb 15 17:50 KICN02-10 - DoH - Students & Industry - Knowledge Triangle Integration.pdf
-rw-r--r--@  1 flow  staff      262208 Feb 15 17:51 KICN02-10 - DoH - Students & Industry - Knowledge Triangle Integration signed.pdf
-rw-r--r--@  1 flow  staff     3341626 Feb 16 08:35 jgs2021-175.pdf
-rw-r--r--@  1 flow  staff       79531 Feb 16 09:12 2021_IRTG_MIP_Second_Phase-2.pdf
-rw-r--r--@  1 flow  staff      796247 Feb 16 15:51 Änderung_Pflieger II[1].pdf
-rw-r--r--@  1 flow  staff      796277 Feb 16 15:51 Änderung_Pflieger.pdf
-rw-r--r--@  1 flow  staff       49208 Feb 16 15:51 Änderungsantrag Nils II.docx
-rw-r--r--@  1 flow  staff      488868 Feb 16 17:22 Akshita_CV copy.pdf
-rw-r--r--@  1 flow  staff           9 Feb 17 08:37 2021_IRTG_MIP_Second_Phase-3.pdf
-rw-r--r--@  1 flow  staff           9 Feb 17 08:37 2021_IRTG_MIP_Second_Phase-4.pdf
-rw-r--r--@  1 flow  staff      859917 Feb 17 08:44 2021_IRTG_MIP_Second_Phase-5.pdf
-rw-r--r--@  1 flow  staff         297 Feb 17 11:16 scholar.enw-7.ris
-rw-r--r--@  1 flow  staff         297 Feb 17 11:17 scholar.enw-8.ris
-rw-r--r--@  1 flow  staff         343 Feb 17 11:21 scholar.ris
-rw-r--r--@  1 flow  staff     2298320 Feb 17 11:23 Hillier2021_Article_Three-DimensionalStructuralGeo.pdf
-rw-r--r--@  1 flow  staff     3967883 Feb 17 11:23 Degen2022_Article_Crustal-scaleThermalModelsRevi.pdf
-rw-r--r--@  1 flow  staff     3967883 Feb 17 11:23 Degen2022_Article_Crustal-scaleThermalModelsRevi-2.pdf
-rw-r--r--@  1 flow  staff        2019 Feb 17 11:24 10.1007%2Fs12665-022-10202-5-citation.ris
-rw-r--r--@  1 flow  staff   173562649 Feb 17 11:28 JabRef-5.5.dmg
-rw-r--r--@  1 flow  staff       72631 Feb 17 11:29 Belegliste CGRE_JARA 2021_korrigiert.pdf
-rw-r--r--@  1 flow  staff      193230 Feb 17 11:29 Belegliste CGRE_JARA 2021_korrigiert signed.pdf
-rw-r--r--@  1 flow  staff      855541 Feb 17 11:58 2021_IRTG_MIP_Second_Phase-6.pdf
-rw-r--r--@  1 flow  staff      796247 Feb 17 17:32 Änderung_Pflieger II.pdf
-rw-r--r--@  1 flow  staff      796247 Feb 17 17:32 Änderung_Pflieger II[2].pdf
-rw-r--r--@  1 flow  staff      796277 Feb 17 17:32 Änderung_Pflieger[1].pdf
-rw-r--r--@  1 flow  staff      718573 Feb 17 17:32 Änderung_PSP-Elemente Leesmeister IV Planstelle ehem. Klinghardt[1].pdf
-rw-r--r--@  1 flow  staff       28515 Feb 17 17:32 Arbeitszeiterhoehung Leesmeister[1].docx
-rw-r--r--@  1 flow  staff       49208 Feb 17 17:32 Änderungsantrag Nils II[1].docx
-rw-r--r--@  1 flow  staff      824323 Feb 17 17:39 Änderung_PSP-Elemente Leesmeister IV Planstelle ehem. Klinghardt[1] signed.pdf
-rw-r--r--@  1 flow  staff      918947 Feb 17 17:40 Änderung_Pflieger[1] signed.pdf
-rw-r--r--@  1 flow  staff      918945 Feb 17 17:40 Änderung_Pflieger II[2] signed.pdf
-rw-r--r--@  1 flow  staff      127196 Feb 17 17:43 Änderungsantrag Nils II[1].pdf
-rw-r--r--@  1 flow  staff      231033 Feb 17 17:43 Änderungsantrag Nils II[1] signed.pdf
-rw-r--r--@  1 flow  staff      133904 Feb 17 17:44 Arbeitszeiterhoehung Leesmeister[1].pdf
-rw-r--r--@  1 flow  staff      215216 Feb 17 17:45 Arbeitszeiterhoehung Leesmeister[1] signed.pdf
-rw-r--r--@  1 flow  staff      106172 Feb 17 17:56 Kalkulationsschema GeoBlocks Jahr 1-3_CGRE.xlsx  [Schreibgeschützt].pdf
-rw-r--r--@  1 flow  staff      219551 Feb 17 17:57 Kalkulationsschema GeoBlocks Jahr 1-3_CGRE.xlsx  [Schreibgeschützt] signed.pdf
-rw-r--r--@  1 flow  staff      287095 Feb 17 17:58 Kalkulationsschema GeoBlocks Jahr 1-3_CGRE.xlsx  [Schreibgeschützt] signed 2.pdf
-rw-r--r--@  1 flow  staff      776074 Feb 17 18:00 20220216 Änderung PSP Ahrensmeier 01-12_2021.pdf
-rw-r--r--@  1 flow  staff      894941 Feb 17 18:02 20220216 Änderung PSP Ahrensmeier 01-12_2021 signed.pdf
-rw-r--r--@  1 flow  staff      730699 Feb 18 08:31 Änderung_PSP-Elemente Leesmeister IV Planstelle ehem. Klinghardt.pdf
-rw-r--r--@  1 flow  staff       28481 Feb 18 08:31 Arbeitszeiterhoehung Leesmeister.docx
-rw-r--r--@  1 flow  staff      135970 Feb 18 08:32 Arbeitszeiterhoehung Leesmeister.pdf
-rw-r--r--@  1 flow  staff      217236 Feb 18 08:33 Arbeitszeiterhoehung Leesmeister signed.pdf
-rw-r--r--@  1 flow  staff      824463 Feb 18 08:34 Änderung_PSP-Elemente Leesmeister IV Planstelle ehem. Klinghardt signed.pdf
-rw-r--r--@  1 flow  staff     5992078 Feb 18 08:36 Timesheets Chudalla Nov21-Jan22.pdf
-rw-r--r--@  1 flow  staff     6223499 Feb 18 08:42 Timesheets Chudalla Nov21-Jan22 2.pdf
-rw-r--r--@  1 flow  staff     2167229 Feb 18 12:04 IMG_0490.HEIC
-rw-r--r--@  1 flow  staff     2128420 Feb 18 15:00 IMG_0493.heic
-rw-r--r--@  1 flow  staff     2042686 Feb 18 15:10 IMG_0497.HEIC
-rw-r--r--@  1 flow  staff     2042686 Feb 18 15:10 IMG_0497 2.HEIC
-rw-r--r--@  1 flow  staff     1985064 Feb 18 15:10 IMG_0501.heic
-rw-r--r--@  1 flow  staff     2685611 Feb 18 15:18 IMG_0505.HEIC
-rw-r--r--@  1 flow  staff     2758505 Feb 18 15:19 IMG_0506.HEIC
-rw-r--r--@  1 flow  staff     1738844 Feb 18 15:24 IMG_0507.HEIC
-rw-r--r--   1 flow  staff      105776 Feb 18 17:37 IMG_0490 copy.jpg
-rw-r--r--   1 flow  staff      109006 Feb 18 17:37 IMG_0507 copy.jpg
-rw-r--r--   1 flow  staff       71413 Feb 18 17:37 IMG_0493 copy.jpg
-rw-r--r--   1 flow  staff      107107 Feb 18 17:37 IMG_0497 2 copy.jpg
-rw-r--r--   1 flow  staff      107107 Feb 18 17:37 IMG_0497 copy.jpg
-rw-r--r--   1 flow  staff       76123 Feb 18 17:37 IMG_0501 copy.jpg
-rw-r--r--   1 flow  staff       90958 Feb 18 17:37 IMG_0506 copy.jpg
-rw-r--r--   1 flow  staff       89774 Feb 18 17:37 IMG_0505 copy.jpg
-rw-r--r--   1 flow  staff     3810925 Feb 18 17:38 IMG_0497 2 copy-1.jpg
-rw-r--r--   1 flow  staff     2934606 Feb 18 17:38 IMG_0493 copy-1.jpg
-rw-r--r--   1 flow  staff     3362821 Feb 18 17:38 IMG_0507 copy-1.jpg
-rw-r--r--   1 flow  staff     4441392 Feb 18 17:38 IMG_0506 copy-1.jpg
-rw-r--r--   1 flow  staff     4399180 Feb 18 17:38 IMG_0505 copy-1.jpg
-rw-r--r--   1 flow  staff     2929642 Feb 18 17:38 IMG_0501 copy-1.jpg
-rw-r--r--   1 flow  staff     3810925 Feb 18 17:38 IMG_0497 copy-1.jpg
-rw-r--r--   1 flow  staff     4030224 Feb 18 17:38 IMG_0490 copy-1.jpg
-rw-r--r--   1 flow  staff      461082 Feb 18 17:39 IMG_0497 2 copy-2.jpg
-rw-r--r--@  1 flow  staff      313573 Feb 18 17:39 IMG_0493 copy-2.jpg
-rw-r--r--@  1 flow  staff      437714 Feb 18 17:39 IMG_0507 copy-2.jpg
-rw-r--r--   1 flow  staff      426122 Feb 18 17:39 IMG_0506 copy-2.jpg
-rw-r--r--   1 flow  staff      426080 Feb 18 17:39 IMG_0505 copy-2.jpg
-rw-r--r--@  1 flow  staff      336496 Feb 18 17:39 IMG_0501 copy-2.jpg
-rw-r--r--@  1 flow  staff      461082 Feb 18 17:39 IMG_0497 copy-2.jpg
-rw-r--r--   1 flow  staff      479390 Feb 18 17:39 IMG_0490 copy-2.jpg
-rw-r--r--@  1 flow  staff    29892360 Feb 21 14:31 Zoom.pkg
-rw-r--r--@  1 flow  staff     2221283 Feb 21 19:54 Geodynamic_modelling_of_the_Century_depo.pdf
-rw-r--r--@  1 flow  staff       42924 Feb 22 17:23 Tagesprogramm_Update_2022-02-11.pdf
-rw-r--r--@  1 flow  staff     6610067 Feb 22 17:57 geo-2019-0614.1.pdf
-rw-r--r--@  1 flow  staff     2756292 Feb 23 06:31 HadCRUT4_accepted.pdf
-rw-r--r--@  1 flow  staff   301990035 Feb 23 09:02 IMG_0579.MOV
-rw-r--r--@  1 flow  staff   179096302 Feb 23 09:12 IMG_0580.MOV
-rw-r--r--@  1 flow  staff    21704740 Feb 23 11:53 HadCRUT.4.6.0.0.median.nc
-rw-r--r--@  1 flow  staff      401568 Feb 23 12:30 Abrechnung Lehrauftrag Budny.pdf
-rw-r--r--@  1 flow  staff      524883 Feb 23 12:40 Abrechnung Lehrauftrag Budny signed.pdf
-rw-r--r--@  1 flow  staff      164648 Feb 23 12:40 Kopie von Fragebogen RST Urlaub+Überstunden JA 2021 -FAK.pdf
-rw-r--r--@  1 flow  staff      281970 Feb 23 12:41 Kopie von Fragebogen RST Urlaub+Überstunden JA 2021 -FAK signed.pdf
-rw-r--r--@  1 flow  staff      195643 Feb 23 12:43 Verpflichtungserklärung_Datencockpit.pdf
-rw-r--r--@  1 flow  staff      338621 Feb 23 12:45 Verpflichtungserklärung_Datencockpit signed Florian Wellmann.pdf
-rw-r--r--@  1 flow  staff    58928466 Feb 23 14:51 IMG_0580_sub.mov
-rw-r--r--@  1 flow  staff       69536 Feb 23 15:44 Bestellung+von+Auftragsscheinen Feb2022.pdf
-rw-r--r--@  1 flow  staff      182547 Feb 23 15:45 Bestellung+von+Auftragsscheinen Feb2022 signed.pdf
-rw-r--r--@  1 flow  staff       21212 Feb 24 11:12 Brisson_Abstract for IAMG '22.docx
-rw-r--r--@  1 flow  staff       21219 Feb 24 11:12 Brisson_Abstract for IAMG 22 FW.docx
-rw-r--r--@  1 flow  staff    50403650 Feb 24 12:02 20220223_093611.mp4
-rw-r--r--@  1 flow  staff    10811086 Feb 24 12:08 Thesis.zip
drwx------@ 11 flow  staff         352 Feb 24 12:08 Thesis
-rw-r--r--@  1 flow  staff       34485 Feb 25 09:50 Verwendungsplanung_DFG_ThinkALPS_2022.pdf
-rw-r--r--@  1 flow  staff      154825 Feb 25 09:52 Verwendungsplanung_DFG_ThinkALPS_2022 signed.pdf
-rw-r--r--@  1 flow  staff       34786 Feb 25 09:53 20220224 Übernahmeantrag_Cremer.pdf
-rw-r--r--@  1 flow  staff      225295 Feb 25 09:53 20220224 Arbeitsmedizinische+Vorsorge_Cremer.pdf
-rw-r--r--@  1 flow  staff       13854 Feb 25 09:53 20220224 Tätigkeitsbericht_Cremer.pdf
-rw-r--r--@  1 flow  staff      154288 Feb 25 09:54 20220224 Übernahmeantrag_Cremer signed.pdf
-rw-r--r--@  1 flow  staff      333956 Feb 25 09:55 20220224 Arbeitsmedizinische+Vorsorge_Cremer signed.pdf
-rw-r--r--@  1 flow  staff     4490956 Feb 25 10:04 geosciences-09-00469-v2.pdf
-rw-r--r--@  1 flow  staff     1468422 Feb 25 10:26 ges02455-2.pdf
-rw-r--r--@  1 flow  staff       35816 Feb 25 10:32 Application+§12.docx
-rw-r--r--@  1 flow  staff       22742 Feb 25 10:32 Notice+of+Readiness+2021.docx
-rw-r--r--@  1 flow  staff       28245 Feb 25 10:38 Betreuungsvereinbarung_engl_PromO21.docx
-rw-r--r--@  1 flow  staff       68206 Feb 25 10:43 Gastzugänge_cgre-guest_20220225.pdf
-rw-r--r--@  1 flow  staff      385317 Feb 25 11:22 AG114318[2].pdf
-rw-r--r--@  1 flow  staff      682195 Feb 25 11:22 Scan20220223163311.pdf
-rw-r--r--@  1 flow  staff      805523 Feb 25 11:23 Scan20220223163311 signed.pdf
-rw-r--r--@  1 flow  staff      503579 Feb 25 11:26 AG114318[2] signed.pdf
-rw-r--r--@  1 flow  staff       62405 Feb 28 20:21 Note 28. Feb 2022.pdf
-rw-r--r--@  1 flow  staff     1529350 Feb 28 22:24 geosciences-11-00150-v2.pdf
-rw-r--r--@  1 flow  staff     4488399 Feb 28 22:35 https___portal.smart-abstract.com_uploads_DGG2022_abstracts_pdf_A-297-ePoster-List-1621d3ffa90b7a.pdf
-rw-r--r--@  1 flow  staff      970010 Mar  1 09:14 2021_IRTG_MIP_Second_Phase-7.pdf
-rw-r--r--@  1 flow  staff     1465430 Mar  1 09:28 220223_FARMIN 2021_Do.pdf
-rw-r--r--@  1 flow  staff     1856777 Mar  1 09:32 220223_FARMIN 2021_Do_signed.pdf
-rw-r--r--@  1 flow  staff       55395 Mar  1 10:27 220224_6. EIT IE CFS 2021 - Representation Letter (004) für Institute_Do.pdf
-rw-r--r--@  1 flow  staff      178862 Mar  1 10:28 220224_6. EIT IE CFS 2021 - Representation Letter (004) für Institute_Do_signed.pdf
-rw-r--r--@  1 flow  staff      525434 Mar  1 10:38 IMG_F1CADD5E6651-1.jpeg
-rw-r--r--@  1 flow  staff     1482151 Mar  1 10:41 Weiterbeschäftigung_0322-0622.pdf
-rw-r--r--@  1 flow  staff     1544030 Mar  1 10:42 Weiterbeschäftigung_0322-0622 signed FW.pdf
-rw-r--r--@  1 flow  staff     3005412 Mar  1 13:42 1-s2.0-S009830042200022X-main.pdf
-rw-r--r--@  1 flow  staff     2298320 Mar  1 17:22 Hillier2021_Article_Three-DimensionalStructuralGeo-2.pdf
-rw-r--r--@  1 flow  staff     6975861 Mar  2 10:59 Mahmoodpour et al (2022)_Energy_Simulations and global sensitivity analysis of THM.pdf
-rw-r--r--@  1 flow  staff      682195 Mar  2 11:02 Scan20220223163311[1].pdf
-rw-r--r--@  1 flow  staff      805820 Mar  2 11:03 Scan20220223163311[1] signed.pdf
-rw-r--r--@  1 flow  staff       62087 Mar  2 15:06 Bewerten Sie Ihr Produkt: Latitude 5420 - (BB) i3-1125G4, i5-1135G7, i7-1165G7.pdf
-rw-r--r--@  1 flow  staff      379468 Mar  2 15:56 Mobility Agreement_STT_2021 Leesmeister.pdf
-rw-r--r--@  1 flow  staff      411202 Mar  2 15:57 Mobility Agreement_STT_2021 Leesmeister sgined.pdf
-rw-r--r--@  1 flow  staff      411202 Mar  2 15:57 Mobility Agreement_STT_2021 Leesmeister signed.pdf
-rw-r--r--@  1 flow  staff       60531 Mar  2 16:08 Precision 3561 zusammenstellen.pdf
drwx------@ 11 flow  staff         352 Mar  2 16:20 Thesis 2
-rw-r--r--@  1 flow  staff       23829 Mar  2 16:55 Notice+of+Readiness+2021 (1).docx
-rw-r--r--@  1 flow  staff     3738680 Mar  2 17:00 d41586-022-00563-z.pdf
-rw-r--r--@  1 flow  staff     1392751 Mar  2 17:57 220120_YOVI_CallforPosters-1.png
-rw-r--r--@  1 flow  staff    26147542 Mar  3 14:58 Zwischenbericht_Teilgebiete_barrierefrei.pdf
-rw-r--r--@  1 flow  staff      143333 Mar  3 14:58 20210706_Corrigenda_Zwischenbericht_Teilgebiete_barrierefrei.pdf
-rw-r--r--@  1 flow  staff     1901650 Mar  3 14:58 Zusammenfassung_Zwischenbericht_Teilgebiete_barrierefrei.pdf
-rw-r--r--@  1 flow  staff     1945375 Mar  3 14:58 Summary_Sub-areas_Interim_Report_barrierefrei.pdf
-rw-r--r--@  1 flow  staff    26811368 Mar  3 14:58 Zwischenbericht_Teilgebiete_-_Englische_Fassung_barrierefrei.pdf
-rw-r--r--@  1 flow  staff     8431262 Mar  3 14:58 UEbersichten_fuer_die_Bundeslaender_zu_den_Teilgebieten_gemaess____13_StandAG_barrierefrei.pdf
-rw-r--r--@  1 flow  staff        4181 Mar  6 18:43 BAHN_Fahrplan_20220524.ics
-rw-r--r--@  1 flow  staff      682979 Mar  6 18:56 FLT_1_CTNLT332470_0.pdf
-rw-r--r--@  1 flow  staff      259462 Mar  6 18:59 Jahressteuerbescheinigung_20220304.pdf
-rw-r--r--@  1 flow  staff      130593 Mar  6 18:59 Info_Jahressteuerbescheinigung_20220304.pdf
-rw-r--r--@  1 flow  staff     1426756 Mar  7 09:31 WB Yang.pdf
-rw-r--r--@  1 flow  staff     1472493 Mar  7 09:32 WB Yang signed.pdf
-rw-r--r--@  1 flow  staff    38168603 Mar  7 09:53 PanoplyMacOS-5.0.2.dmg
-rw-r--r--@  1 flow  staff     1492828 Mar  7 09:57 WB Ryan Andika.pdf
-rw-r--r--@  1 flow  staff     1541500 Mar  7 09:58 WB Ryan Andika signed.pdf
-rw-r--r--@  1 flow  staff      483953 Mar  7 11:02 108.pdf
-rw-r--r--@  1 flow  staff     1033478 Mar  7 11:02 se-10-1469-2019.pdf
-rw-r--r--@  1 flow  staff      301687 Mar  7 11:14 geoIDG.pdf
-rw-r--r--@  1 flow  staff     8078394 Mar  7 12:04 DGG2022_Tagungsband.pdf
drwxr-xr-x@ 15 flow  staff         480 Mar  7 14:46 zulu17.32.13-ca-jdk17.0.2-macosx_aarch64
drwxr-xr-x@ 11 flow  staff         352 Mar  7 14:47 zulu17.32.13-ca-jre17.0.2-macosx_aarch64
drwxr-xr-x@  6 flow  staff         192 Mar  7 14:50 PanoplyJ
-rw-r--r--@  1 flow  staff       15491 Mar  7 18:12 smime.p7m
-rw-r--r--@  1 flow  staff       50565 Mar  8 07:58 KIC-Publikation_Freigabe (1).pdf
-rw-r--r--@  1 flow  staff       41032 Mar  8 07:58 10_Wellmann -2.docx
-rw-r--r--@  1 flow  staff       40934 Mar  8 08:04 10_Wellmann -2_FW.docx
-rw-r--r--@  1 flow  staff      172947 Mar  8 08:08 KIC-Publikation_Freigabe (1) signed FW.pdf
-rw-r--r--@  1 flow  staff     4488399 Mar  9 08:06 A-297-ePoster-List-1621d3ffa90b7a.pdf
-rw-r--r--@  1 flow  staff     1664922 Mar  9 09:37 had4_krig_ensemble_v2_0_0.txt
-rw-r--r--@  1 flow  staff   318662360 Mar  9 10:17 HadCRUT.5.0.1.0.analysis.anomalies.91_to_100_netcdf.zip
-rw-r--r--@  1 flow  staff      281805 Mar  9 16:40 it.pdf
-rw-r--r--@  1 flow  staff     3179986 Mar 10 08:58 Toolkits_Workshop CGRE_ Gempy-Blender Addon_1.zip
drwxr-xr-x@  6 flow  staff         192 Mar 10 08:58 Toolkits_Workshop CGRE_ Gempy-Blender Addon_1
-rw-r--r--@  1 flow  staff      329962 Mar 10 09:30 20220310 Corona GBU Veranstaltung_31.04.2022.pdf
-rw-r--r--@  1 flow  staff      239009 Mar 10 09:30 20220310 Raumvergabeantrag_MN15_MN16_21.04.2022.pdf
-rw-r--r--@  1 flow  staff      300976 Mar 10 09:31 20220310 Raumvergabeantrag_MN15_MN16_21.04.2022 signed.pdf
-rw-r--r--@  1 flow  staff      448898 Mar 10 09:32 20220310 Corona GBU Veranstaltung_31.04.2022 signed.pdf
-rw-r--r--@  1 flow  staff     1045935 Mar 10 10:03 2021_IRTG_MIP_Second_Phase-8.pdf
-rw-r--r--@  1 flow  staff       71059 Mar 10 17:27 W3_Tektonik und Geodynamik_ZWA[2].docx
-rw-r--r--@  1 flow  staff       42599 Mar 10 18:25 the-data-science-venn-diagram.html
-rw-r--r--@  1 flow  staff       75018 Mar 10 20:36 W3_Tektonik und Geodynamik_ZWA[2]b.docx
-rw-r--r--@  1 flow  staff      397502 Mar 10 21:02 Ertraegnisaufstellung_20220308.pdf
-rw-r--r--@  1 flow  staff      259473 Mar 10 21:07 Jahressteuerbescheinigung_20210309.pdf
-rw-r--r--@  1 flow  staff      396843 Mar 10 21:10 Ertraegnisaufstellung_20210309.pdf
-rw-r--r--@  1 flow  staff      270676 Mar 10 21:11 Direkt_Baufinanzierung_2007663057_Kontoauszug_20220106.pdf
-rw-r--r--@  1 flow  staff      270812 Mar 10 21:12 Direkt_Baufinanzierung_2007663057_Kontoauszug_20210107.pdf
-rw-r--r--@  1 flow  staff     1044620 Mar 11 17:39 ReliableProspectionAndExploration.pdf
-rw-------@  1 flow  staff      334666 Mar 12 11:57 tensor.pdf
-rw-r--r--@  1 flow  staff        2532 Mar 12 12:00 citations-20220312T110005.bibtex
-rw-r--r--@  1 flow  staff        2308 Mar 12 12:00 citations-20220312T110012.enw
-rw-r--r--@  1 flow  staff     1802533 Mar 12 12:00 ges02455-3.pdf
(base) flow@Florians-Air Downloads % vi citations-20220312T110005.bibtex

@article{10.1130/GES02455.1,
    author = {Wellmann, Florian and Virgo, Simon and Escallon, Daniel and de la Varga, Miguel and Jüstel, Alexander and Wagner, Florian M. and Kowalski, Julia and Zhao, Hu and Fehling, Robin and Chen, Qian},
    title = "{Open AR-Sandbox: A haptic interface for geoscience education and outreach}",
    journal = {Geosphere},
    year = {2022},
    month = {02},
    issn = {1553-040X},
    doi = {10.1130/GES02455.1},
    url = {https://doi.org/10.1130/GES02455.1},
    eprint = {https://pubs.geoscienceworld.org/gsa/geosphere/article-pdf/doi/10.1130/GES02455.1/5541527/ges02455.pdf},
}



Feel free to download and use the Open AR-Sandbox software! We do not provide any
warranty and any guarantee for the use. We also do not provide professional
support, but we aim to answer questions posted as Issues on the github page as
quickly as possible.

Open AR-Sandbox is published under an **GNU Lesser General Public License v3.0**, which
means that you are
free to use it, if you do not do any modifications, in a wide variety of ways
(even commercially). However, if you plan to _modify and redistribute_ the
code, you also _have to make it available under the same license_!

Also, if you do any modifications, especially for scientific and educational use,
then please _provide them back to the main project_ in the form of a pull request,
as common practice in the open-source community. If you have questions on
the procedure, feel free to contact us about it.

These are the main conditions for using this library:
- License and copyright notice
- Disclose source
- State changes
- Same license (library)

For more details on the licsense, please see provided license file.

:warning: **Warning!** It is unfortunate that we have to state this here, but: downloading the software and presenting it somewhere as your 
own work is serious **scientific fraud**! And if you develop content further, then please push these developments
back to this repostory - in the very interest of scientific development (and also a requirement of the license).
For more details, please consult the information below and the license.


Requirements
--------
You will need: 
* Microsoft Kinect (we tested the first and second generation kinect with a usb adapter, but every kinect compatible 
with the pykinect drivers will likely work).
* Projector
* A box of Sand

Mount the kinect and projector facing down vertically in the center above of the box. The optimal distance will depend 
on the size of your sandbox and the optics of the projector, from our experience a distance of 150 cm is well suited 
for a 80 cm x 100 cm box. 
More details on how to set up the kinect and projector can be found in the `1_calib_projector.ipynb` 
and `2_calib_sensor.ipynb` notebooks, and if you want to use the ArUco markers `3_calib_arucos.ipynb`.


Installation 
-----
First of all you will need a healthy Python 3 environment. We recommend using 
[Anaconda](https://www.anaconda.com/distribution/). In addition to some standard Python packages, you will need a 
specific setup dependent on the Kinect version you are using. In the following we provide detailed installation 
instructions.\
Now download or clone this repository [open_AR_Sandbox](https://github.com/cgre-aachen/open_AR_Sandbox) from github.

1. First clone the repository:
```
git clone https://github.com/cgre-aachen/open_AR_Sandbox.git
```
2. Enter the new downloaded project folder:
```
cd open_AR_Sandbox
```
3. Create a new anaconda environment
```
conda create -n sandbox-env python=3.8
```
4. Now when you want to use the sandbox and the packages we are about to installl you will have to activate the 
environment before starting anything
```
conda activate sandbox-env
```
### Standard packages

To install all the standard packages please use the  `requirements.txt` file:

```
pip install -r requirements.txt
```

[RECOMMENDED] You can also have a local installation of the sandbox by using the File "setup.py" by doing:

```
pip install -e . 
```

[ALTERNATIVELY] You can use our `sandbox-environment.yml` file to instantly install all the dependencies with the 
extensions. Beware that you  still need to install the kinect sensors by yourself according to your operative system.
```
conda env create -f sandbox_environment.yml
```

### Download sample data

You have the option to download some publicly shared files from our Open AR-Sandbox project shared folder. 
You will need to do this if you want to run the tests, use the landslides simulations and/or get the trained models for 
the use of the Landscape generation module.

In the terminal type:

```
python3 sandbox/utils/download_sample_datasets.py
```

and follow the instruction on the terminal to download the specific files you need. We use 
[Pooch](https://github.com/fatiando/pooch) to help us fetch our data files and store them locally in your computer 
to their respective folders. Running this code a second time will not trigger a download since the file already exists.

You can also follow the Jupyter Notebook ['Download_datasets.ipynb'](notebooks/tutorials/) and follow the commands. 

### Kinect Installation 
 
### For Windows

#### Kinect v1 - Future

There is still no support for kinect V1... 

#### Kinect V2 - PyKinect2

(Tested on Windows 10). First, **install the current** 
[Kinect SDK](https://www.microsoft.com/en-us/download/confirmation.aspx?id=44561) **including drivers**. 
You can use the software bundle to test the connection to your
 kinect, before you continue.

To make Python and the Kinect SDK communicate, install the related [PyKinect2](https://github.com/Kinect/PyKinect2) 
wrappers which can be easily installed via:

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

#### Kinect v1 - libfreenect

To make Open AR-Sandbox talk to the first generation kinect you will need the
[Libfreenect Drivers](https://github.com/OpenKinect/libfreenect) with
[Python Wrappers](https://openkinect.org/wiki/Python_Wrapper). 
The installation is kind of straight forward for Linux and MacOS but 
challenging for Microsoft (in fact: if you pull it off, let us know how you did it!)
The steps can be summarized as follows (refer to any problems regarding installation in to
[link](https://github.com/OpenKinect/libfreenect))
To build libfreenect, you'll need

- [libusb](http://libusb.info) >= 1.0.18 (Windows needs >= 1.0.22)
- [CMake](http://cmake.org) >= 3.12.4 (you can visit 
[this](https://www.claudiokuenzler.com/blog/796/install-upgrade-cmake-3.12.1-ubuntu-14.04-trusty-alternatives)
page for detailed instructions for the installation)

Once these are installed we can follow the next commands
```
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
```

#### Kinect v2 - freenect2
or pylibfreenect2 \
For this we are going to use a python interface for the library 
[libfreenect2](https://github.com/OpenKinect/libfreenect2)
called [freenect2](https://github.com/rjw57/freenect2-python). 
* First we need to install the [freenect2](https://github.com/rjw57/freenect2-python) as described in the installation 
guide. 
The steps can be summarized as follows (refer to any problems regarding installation in to 
[link](https://rjw57.github.io/freenect2-python/))
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
* Set up udev rules for device access: 
```
sudo cp ../platform/linux/udev/90-kinect2.rules /etc/udev/rules.d/
```
Now unplug and replug the Kinect sensor.
* Test if the kinect is correctly installed, by running:
```
./bin/Protonect
```
* You should be able to see the kinect image working. If not, check  [libfreenect2](https://github.com/OpenKinect/libfreenect2) 
installation guide for more detailed instructions of installation

* If everything is working until now, we can install the python wrapper. For this first we need to indicate where the 
`freenect2` folder can be found.
```
export PKG_CONFIG_PATH=$HOME/freenect2/lib/pkgconfig
```
NOTE: if you installed the `freenect2` in other location, specify variables with the corresponding path
* now we can use `pip install` , or any other method described in the [freenect2](https://github.com/rjw57/freenect2-python) 
installation guide. 
```
pip install freenect2
```
IMPORTANT: To this point will work in any python that starts with the terminal. Nevertheless, if we start python from 
another source, the error 
`ImportError: libfreenect2.so.0.2: cannot open shared object file: No such file or directory` will appear every time we 
import the package. To fix this problem we will need
to export the variables again or if you want a more permanent solution, open the `.bashrc` file and paste the following
 at the end of the file:
```
# set PATH to freenect2 to be imported in python
export PKG_CONFIG_PATH=$HOME/freenect2/lib/pkgconfig
```
* With this it will always work for any python open from the terminal. Including jupyter notebooks
* But now if we want to run this package in Pycharm or symilar, we can directly copy the 3 files 
(`libfreenect2.2.s0...`) from the `freenect2/lib` folder into the 
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


### LiDAR L515 Installation 

#### Installing in Windows

First, go to the latest release page on [GitHub](https://github.com/IntelRealSense/librealsense/releases/latest) 
and download and execute the file: 

```Intel.RealSense.Viewer.exe```

Follow the instructions for the installation and update the firmware of your sensor.  You should be able to use and see the depth and RGB image.

#### Installing in Linux 

Detailed installation steps can be found in the 
[linux installation guide](https://github.com/IntelRealSense/librealsense/blob/development/doc/distribution_linux.md). 
The steps are as follows:

- Register the server's public key:  
`sudo apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE`
In case the public key still cannot be retrieved, check and specify proxy settings: `export http_proxy="http://<proxy>:<port>"`  
, and rerun the command. See additional methods in the following [link](https://unix.stackexchange.com/questions/361213/unable-to-add-gpg-key-with-apt-key-behind-a-proxy).  

- Add the server to the list of repositories:  
  Ubuntu 16 LTS:  
`sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo xenial main" -u`  
  Ubuntu 18 LTS:  
`sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo bionic main" -u`  
  Ubuntu 20 LTS:  
`sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo focal main" -u`

- Install the libraries:  
  `sudo apt-get install librealsense2-dkms`  
  `sudo apt-get install librealsense2-utils`  
  
Reconnect the Intel RealSense depth camera and run: `realsense-viewer` to verify the installation.

#### Running with python

After the sensor is installed on your pltaform, the Python wrapper can be easily installed via:

```pip install pyrealsense2```

If any problems with the installation reference to 
[Intel RealSense Python Installation](https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python#installation)


External Packages
---------
### GemPy

To use implicit geological models inside the sandbox, go to [GemPy](https://github.com/cgre-aachen/gempy),
clone or download the repository and follow the Gempy Installation instructions. With gempy installed 
you can follow the tutorial [GempyModule](notebooks/tutorials/04_GempyModule/).
```
pip install gempy
```
If using windows you will need to install `Theano` separately as instructed in [here](https://www.gempy.org/installation)
```
conda install mingw libpython m2w64-toolchain
conda install theano
pip install theano --force-reinstall
```

Optional: 
Gempy will print some output each time a frame is calculated, which can fill up the console. to supress this, go to your gempy installation and comment out line 381 in ```
gempy/core/model.py```

```
# print(f'Active grids: {self._grid.grid_types[self._grid.active_grids]}')
```

### Devito

This package uses the power of [Devito](https://github.com/devitocodes/devito) to run wave proppagation simmulations.
More about this can be found in `notebooks/tutorials/10_SeismicModule/`. Follow the Devito installation instructions. 
* This module so far have only support in linux 
```
pip install --user git+https://github.com/devitocodes/devito.git
```

### PyGimli
This library is a powerful tool for Geophysical inversion and Modelling. Some examples can be found in 
`notebooks/tutorials/11_Geophysics/`. 
[PyGimli](https://www.pygimli.org/) can be installed following the installation intructions 
[here](https://www.pygimli.org/installation.html)

We recommend creating a new environment where PyGimli is already installed and over that one install the sandbox 
dependencies.
```
conda create -n sandbox-env -c gimli -c conda-forge pygimli=1.1.0
```
* And now go back to [installation](README.md#installation) and follow all over again the instruction but skipping 
step 2. 

### PyTorch

To use the LandscapeGeneration module we need to install [PyTorch](https://pytorch.org/). This module use the power 
of [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) 
to take a topography from the sandbox, translate this as a DEM and then display it again on the sandbox as a Landscape 
image. 
To install the dependencies for this module do:
```
#For Windows
pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```
```
#For Linux
pip install torch torchvision
```
```
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix
pip install -r requirements.txt
```
Once this is installed, copy the trained model in `/notebooks/tutorials/09_LandscapeGeneration/checkpoints` folder, 
and then follow the notebook.
Get in contact with us to provide you with the train model for this module. 


Project Development
-------------------

Open AR-Sandbox is developed at the research unit [Computational Geoscience
and Reservoir Engineering (CGRE)](https://www.cgre.rwth-aachen.de/) at RWTH Aachen University, Germany.

[![CGRE](https://www.cgre.rwth-aachen.de/global/show_picture.asp?id=aaaaaaaaabaxmhn)](https://www.cgre.rwth-aachen.de/)



### Project Lead

[Prof. Florian Wellmann, PhD](https://www.cgre.rwth-aachen.de/cms/CGRE/Das-Lehr-und-Forschungsgebiet/~dnyyj/Prof-Wellmann/lidx/1/)

### Maintainers (also external to CGRE)

* Daniel Escallón 
* Simon Virgo
* Miguel de la Varga

Obtaining a full system
-----------------------

If you are interested in buying a fully operating set-up including appropriate
hardware, pre-installed software, and set-up and maintenance, please contact
[Terranigma Solutions GmbH](https://www.terranigma-solutions.com/services).



 
