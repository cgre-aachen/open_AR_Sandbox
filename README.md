# open_AR_Sandbox

##  Introduction
Augmented Reality Sandboxes are a great tool for science outreach and teaching due to their intuitive and interaction-enhancing operation. Recently AR Sandboxes are becoming increasingly popular as interactive exhibition pieces, teaching aids and toys.

AR Sandboxes consist of a box of sand that can be freely sculpted by hand. The topography of the sand is constantly scanned with a depth camera and a computed image is projected back onto the sand surface, augmenting the sandbox with digital information.

However, most of these common AR Sandboxes are limited to the visualization of topography with contour lines and colors, as well as water simulations on the digital terrain surface. The potential for AR Sandboxes for geoscience education , and especially for teaching struc- tural geology, remains largely untapped.

For this reason, we have developed open-AR-Sandbox, an augmented reality sandbox designed specifically for the use in geoscience education. In addition to the visualization of topography it can display geologic subsurface information such as the outcropping lithology, creating a dynamic and interactive geological map. The relations of subsurface structures, topography and outcrop can be explored in a playful and comprehensible way.

## Requirements
You will need: 
* Microsoft Kinect (we tested first generation kinect with a usb adapter, but every kinect compatible with the freenect drivers will likely work)
* Projector
* A box of Sand 


Mount the kinect and projector facing down vertically in the center above of the box. The optimal distance will depend on the size of your sandbox and the optics of the projector, from our experience a distance of 150 cm is well suited for a 80 cm x 100 cm box. 
More details on how to set up the kinect and projector can be found in the `calibrate sandbox.ipynb` notebook
## Installation 
To make the sandbox run you need two libraries and their dependencies: [open_AR_Sandbox](https://github.com/cgre-aachen/open_AR_Sandbox) and [GemPy](https://github.com/cgre-aachen/gempy)
clone or download both repositories and follow the Gempy Installation instructions. 

To make open_AR_Sandbox talk to the kinect you will need the [Libfreenect Drivers](https://github.com/OpenKinect/libfreenect) with [Python Wrappers] (https://openkinect.org/wiki/Python_Wrapper). The installation is kind of straight forward for Linux and MacOS but challenging for Microsoft (in fact: if you pull it off, let us know how you did it!)
  

## Getting started
So now the necessary software is installed and (hopefully) running and you have set up your Sandbox with a projector and a kinect, it is time to calibrate your sandbox.
The calibration step is necessary to align your sandbox with the kinect as well as with the projected image. 

Navigate to the Folder `/notebooks/tutorials` and follow the instruction in the `calibrate sandbox.ipynb` notebook. 
If everything goes right you will end up wit a calibration file that you can use for your Sandbox setup.

The next (and most exciting) step is to make the sandbox run and actually showing some geology: the `run geologic model.ipynb` notebook will walk you through the necessary steps. 


