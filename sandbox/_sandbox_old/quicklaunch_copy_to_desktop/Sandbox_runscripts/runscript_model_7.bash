#!/bin/bash

MODEL='/home/sarndbox/GempySandbox/gempy/notebooks/examples/Models_Lisa/lisa-7.pickle'

#Extent:

XMIN=0.0
XMAX=2000.0

YMIN=0.0
YMAX=2000.0

ZMIN=300.0
ZMAX=1633.0

echo $XMIN $XMAX $YMIN $YMAX $ZMIN $ZMAX
CALIBRATION='/home/sarndbox/GempySandbox/open_AR_Sandbox/notebooks/tutorials/sandbox_brisbane.dat'

WORKDIR='/home/sarndbox/GempySandbox/open_AR_Sandbox/notebooks/tutorials/'

cd $WORKDIR
pwd

/home/sarndbox/anaconda3/bin/python run_geologic_model.py $MODEL $CALIBRATION $XMIN $XMAX $YMIN $YMAX $ZMIN $ZMAX || echo "There is probably a script still running that is blocking the kinect. please close all other sandbox programs first! If nothing helps try to unplug and reconnect the kinect usb cable"

