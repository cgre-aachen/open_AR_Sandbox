.. AR_Sandbox documentation master file, created by
   sphinx-quickstart on Tue Apr 14 17:11:54 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Modules
=======

The open_AR_Sandbox as well as `GemPy <https://www.gempy.org/>`_ are under continuous development and including more
modules for major outreach.

Implemented modules
~~~~~~~~~~~~~~~~~~~

- MarkerDetection: Place virtual boreholes in the model, Define a cross section with multiple markers, Set the start
  position for simulations (landslides, earthquakes, etc.) check the `ArUcos marker detection <https://docs.opencv.org/trunk/d5/dae/tutorial_aruco_detection.html>`_ for more information
- TopoModule: Normalize the depth image to display a topography map with fully customizable contour lines and variable
  heights
- SearchMethodsModule: Takes the depth image and performs Monte-Carlo simulation algorithms to construct the probability
  distribution based on the structure of the current DEM in an interactive way
  (https://chi-feng.github.io/mcmc-demo/app.html)
- `GemPy <https://www.gempy.org/>`_ Module: Use the full advantage of the powerful `GemPy <https://www.gempy.org/>`_
  package to construct geological models and visualize them on the sandbox in real-time
- GradientModule: Takes the gradient information from the depth image and highlight slopes in x and y direction,
  calculation of laplacian, interactive hill shading, visualization of a vector field, and streamline plot
- LoadSaveTopoModule: Takes the depth image and allows it to be saved as a DEM to reconstruct topographies previously
  constructed
- LandslideSimulation: With precomputed landslides simulations, recreate a topography and trigger a landslide to
  visualize its flow, direction, and velocity in real-time, or frame by frame
- PrototypingModule: Create your own module with the help of this module to link the live threading of the sandbox with
  your ideas
- LandscapeModule: Landscape generations using machine learning codes powered by CycleGAN
- SeismicModule: module for seismic wave modelling in the sandbox. This uses the power of Devito
- GeoelectricsModule: module for visualization of geoelectrical fields using aruco markers as electrodes. This use power
  of PyGimli


Modules in implementation process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- More Tutorials, examples, tests and documentation to help you develop your own modules
- `GemPy <https://www.gempy.org/>`_ optimization for (much!) higher frame-rates
- On-the-fly modification of the geological model (layer dip, thickness fault throw, etc.)
- Integration of more depth sensors (support to all kinect sensors)
- Improve compatibility with Linux and MacOS
- ...