

# **Fish 3D Size Estimation with Monocular Camera using Computer Vision and Machine Learning Techniques**

A complete pipeline of Object detection, 3D detection, size estimation for fish using Computer Vision and Machin elearning. The goal of this project is to use machine learning and computer vision techniques to estimate fish sizes in 3D from monocular video footage

![Pipeline](https://github.com/alfinnurhalim/centernet_3d_fish/blob/master/readme/Pipeline.png)

## Highlight

The pipeline composed by 4 different Module:
- **YOLO:** The base detector for 2D object Detection. it's the first model to be run, the output is 2D bbox of the detected fish. YOLO v8 is used in this repo. the pretrained model available in [INSTALL.md](readme/INSTALL.md) 

- **ByteTrack:** The tracking module of the 2D detector, it used to keep track on the fish detected by YOLO. the output of this model is and ID for every 2D box detected.

- **CenterNet:** The main model to predict the size of the fish in 3D space. it predict the dimension of the fish (in meters) and depth. 
- **Association**: The output of YOLO is more robust compared to CenterNet. but YOLO does not predict 3D property. this module basically associating the YOLO result with the CenterNet 3d result to get the 3D properties. the Association is based on IoU matching of Bounding Box of both YOLO and CenterNet

- **Sim2Real**:To create a better quality dataset, we enhanced it using an image-to-image generative model known as sim2real. This model essentially functions as a style-transfer mechanism, stylizing input images based on a reference style image. The setup of sim2real can be seen [here](https://github.com/neuralxmasaki/sim_to_real)

## Usage
### Inference
To run the model simply run
```shell
%cd src
python show_video.py -i <input video>
```
Digging more into the code. the inference code `show_video.py`consist of 3 section

-  YOLO Model
 To load YOLO model simply add following code. The output is a DataFrame in KITTI format

```python
    import inference_yolo as YOLO
    YOLO_df = YOLO.forward(img_dirs,max_img)
```
-  CenterNet Model
 Similar to YOLO, to load and run CenterNet model simply add following code. the output is also in KITTI format

```python
    import inference_centernet as CenterNet
    centernet_df = CenterNet.forward(img_dirs,max_img)
```
-  Association Module
Association module take 2 DataFrame, and the output is a new DataFrame of the result of the association. the post processing module also included in the Association module.

```python
    centernet_df = CenterNet.forward(img_dirs,max_img)
    YOLO_df = YOLO.forward(img_dirs,max_img)
	
    associated_df = match(centernet_df,YOLO_df)
```
-  Painter Module
Painter basically a Class for any painting operation. 

```python
'''
AVAILABLE DRAWING FUNCTION

3d bbox			--> painter.draw_3d_box()
2d bbox			--> painter.draw_2d_box()
dashed 2d bbox	--> painter.draw_dashed_2d_box()
fish size label --> painter.draw_size_label()
fish count 		--> painter.draw_fish_count(len(data))
frame number	--> painter.draw_frame_num(int(idx))
'''
from Painter import Painter

painter = Painter(name=name)

painter.set_img(img)
painter.set_label(label)

painter.draw_3d_box()
painter.draw_dashed_2d_box()
painter.draw_size_label()
painter.draw_fish_count(len(data))

# to save the ouput as video, simply call update after every image iteration. then save at the end 
painter.update()
painter.save_video()
```
	

REALLY RECOMENNED to explore the `show_video.py` as it's really easy to understand and the code is clean enough
## Installation

Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.

## Dataset
The dataset employed is in a 3D object region proposal format, inspired by the KITTI dataset. Each data entry comprises an RGB image paired with its annotation.
