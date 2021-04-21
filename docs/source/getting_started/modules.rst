.. AR_Sandbox documentation master file, created by
   sphinx-quickstart on Tue Apr 14 17:11:54 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Modules
=======

The `open_AR_Sandbox <https://github.com/cgre-aachen/open_AR_Sandbox>`_ as well as `GemPy <https://www.gempy.org/>`_ are
under continuous development and including more modules for major outreach.

Implemented modules
~~~~~~~~~~~~~~~~~~~

- `MarkerDetection <https://github.com/cgre-aachen/open_AR_Sandbox/tree/master/notebooks/tutorials/00_Calibration>`_:
  Place virtual boreholes in the model, define a cross section with multiple markers, set the start
  position for simulations (landslides, earthquakes, etc.). For more information check
  `ArUco's marker detection <https://docs.opencv.org/trunk/d5/dae/tutorial_aruco_detection.html>`_
- `TopoModule <https://github.com/cgre-aachen/open_AR_Sandbox/tree/master/notebooks/tutorials/02_TopoModule>`_:
  Normalize the depth image to display a topography map with fully customizable contour lines and variable
  heights
- `SearchMethodsModule <https://github.com/cgre-aachen/open_AR_Sandbox/tree/master/notebooks/tutorials/03_SearchMethodsModule>`_:
  Takes the depth image and performs Monte-Carlo simulation algorithms to construct the probability distribution based
  on the structure of the current DEM in an interactive way
  (`Hamiltonian Monte Carlo demo <https://chi-feng.github.io/mcmc-demo/app.html>`_)
- `GemPyModule <https://github.com/cgre-aachen/open_AR_Sandbox/tree/master/notebooks/tutorials/04_GempyModule>`_: Use
  the full advantage of the powerful `GemPy <https://www.gempy.org/>`_ package to construct geological models and
  visualize them on the sandbox in real-time
- `GradientModule <https://github.com/cgre-aachen/open_AR_Sandbox/tree/master/notebooks/tutorials/05_GradientModule>`_:
  Takes the gradient information from the depth image and highlight slopes in x and y direction, calculation of
  laplacian, interactive hill shading, visualization of a vector field, and streamline plot
- `LoadSaveTopoModule <https://github.com/cgre-aachen/open_AR_Sandbox/tree/master/notebooks/tutorials/06_LoadSaveTopoModule>`_:
  Takes the depth image and allows it to be saved as a DEM to reconstruct topographies previously constructed
- `LandslideSimulation <https://github.com/cgre-aachen/open_AR_Sandbox/tree/master/notebooks/tutorials/07_LandslideSimulation>`_:
  With precomputed landslides simulations, recreate a topography and trigger a landslide to visualize its flow,
  direction, and velocity in real-time, or frame by frame
- `PrototypingModule <https://github.com/cgre-aachen/open_AR_Sandbox/tree/master/notebooks/tutorials/08_PrototypingModule>`_:
  Create your own module with the help of this module to link the live threading of the sandbox with your ideas
- `LandscapeModule <https://github.com/cgre-aachen/open_AR_Sandbox/tree/master/notebooks/tutorials/09_LandscapeGeneration>`_:
  Landscape generations using machine learning codes powered by `CycleGAN <https://junyanz.github.io/CycleGAN/>`_
- `SeismicModule <https://github.com/cgre-aachen/open_AR_Sandbox/tree/master/notebooks/tutorials/10_SeismicModule>`_:
  Module for seismic wave modelling in the sandbox. This uses the power of `Devito <https://www.devitoproject.org/>`_
- `GeoelectricsModule <https://github.com/cgre-aachen/open_AR_Sandbox/tree/master/notebooks/tutorials/11_GeoelectricsModule>`_:
  Module for visualization of geoelectrical fields using
  `ArUco <https://docs.opencv.org/trunk/d5/dae/tutorial_aruco_detection.html>`_ markers as electrodes. This use power
  of `PyGimli <https://www.pygimli.org/>`_

Check the video below for some of the features in action:

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/watch?v=t0fyPVMIH4g" frameborder="0" allowfullscreen style="position: absolute;
         top: 0; left: 0; width: 100%; height: 90%; margin-bottom: 2em;"></iframe>
    </div>

Modules in implementation process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- More Tutorials, examples, tests and documentation to help you develop your own modules
- `GemPy <https://www.gempy.org/>`_ optimization for (much!) higher frame-rates
- On-the-fly modification of the geological model (layer dip, thickness fault throw, etc.)
- Integration of more depth sensors (support to all kinect sensors)
- Improve compatibility with Linux and MacOS
- ...