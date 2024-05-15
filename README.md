# SampleBridges-MILLanD2023
This is the code to the MICCAI 2023 workshop MILLanD paper: "Using Training Samples as Transitive Information Bridges in Predicted 4D MRI" https://link.springer.com/chapter/10.1007/978-3-031-44917-8_23

Train a model
==================
train.py is used to train a new model

Data
==================
The data must be in nifti or nrrd file format

Prediction
==================
Use predict.py to predict a series of 3D MRI volumes based of of an reference sequence of a subject.