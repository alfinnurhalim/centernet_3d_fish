## Installation

The Pipelin ewas devided into 3 different mode. YOLO for 2D object detection, CenterNet for 3d object detection, and Bytetrack for object tracking

1. Create the conda envrionment using `environment.yml`:

```shell
conda env create -f environment.yml
```

2. Install the required Python packages from the `requirements.txt` file using pip:
```shell
pip install -r requirements.txt  
```
3. Download the pre-trained model and put it in 'models', update the model path in `inference_yolo.py` and `inference_centernet.py` to specify the correct path
```shell
#YOLO
https://drive.google.com/file/d/1OBY4tta4LefgEjpvjMzwoydJB5-0XLQJ/view?usp=sharing

#CenterNet
https://drive.google.com/file/d/1H5RMScxTRgsPVhqa1nwprMlbma25u7SI/view?usp=sharing
```

4. Install DCNv2

```shell
cd src/lib/models/networks/DCNv2
rm -rf build
python setup.py build develop
```

5. Install external modules

```shell
cd src/lib/external
make
```

## Usage
### Local

```shell
python show_video.py -i <input video>
```