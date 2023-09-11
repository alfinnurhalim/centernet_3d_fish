
# **Fish 3D Size Estimation with Monocular Camera using Computer Vision and Machine Learning Techniques**

A complete pipeline of Object detection, 3D detection, size estimation for fish using Computer Vision and Machin elearning. The goal of this project is to use machine learning and computer vision techniques to estimate fish sizes in 3D from monocular video footage

## Updates

 - (June, 2020) We released a state-of-the-art Lidar-based 3D detection and tracking framework [CenterPoint](https://github.com/tianweiy/CenterPoint).
 - (April, 2020) We released a state-of-the-art (multi-category-/ pose-/ 3d-) tracking extension [CenterTrack](https://github.com/xingyizhou/CenterTrack).

## Highlight

The pipeline composed by 4 different Module:
- **YOLO:** The base detector for 2D object Detection. it's the first model to be run, the output is 2D bbox of the detected fish. YOLO v8 is used in this repo. the pretrained model available in [INSTALL.md](readme/INSTALL.md) 

- **ByteTrack:** The tracking module of the 2D detector, it used to keep track on the fish detected by YOLO. the output of this model is and ID for every 2D box detected.

- **CenterNet:** The main model to predict the size of the fish in 3D space. it predict the dimension of the fish (in meters) and depth. 
- **Association**: The output of YOLO is more robust compared to CenterNet. but YOLO does not predict 3D property. this module basically associating the YOLO result with the CenterNet 3d result to get the 3D properties. the Association is based on IoU matching of Bounding Box of both YOLO and CenterNet

- **Sim2Real**:To create a better quality dataset, we enhanced it using an image-to-image generative model known as sim2real. This model essentially functions as a style-transfer mechanism, stylizing input images based on a reference style image. The setup of sim2real can be seen [here](https://github.com/neuralxmasaki/sim_to_real)


## Installation

Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.

## Dataset
The dataset employed is in a 3D object region proposal format, inspired by the KITTI dataset. Each data entry comprises an RGB image paired with its annotation.
