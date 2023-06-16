from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm
import pickle
import json
import numpy as np
import cv2
DATA_PATH = '../../data/kitti/'
DEBUG = False
# VAL_PATH = DATA_PATH + 'training/label_val/'
import os
import pandas as pd
SPLITS = ['3dop'] 
import _init_paths
from utils.ddd_utils import compute_box_3d, project_to_image, alpha2rot_y
from utils.ddd_utils import draw_box_3d, unproject_2d_to_3d

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

'''
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
'''

def _bbox_to_coco_bbox(bbox):
  return [(bbox[0]), (bbox[1]),
          (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]

def read_clib(calib_path):
  f = open(calib_path, 'r')
  for i, line in enumerate(f):
    if i == 2:
      calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
      calib = calib.reshape(3, 4)
      return calib

cats = ['Car', 'DontCare']
cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}
# cat_info = [{"name": "pedestrian", "id": 1}, {"name": "vehicle", "id": 2}]
F = 210
H = 384 # 375
W = 384 # 1242
EXT = [0,0,0]
CALIB = np.array([[F, 0, W / 2, EXT[0]], [0, F, H / 2, EXT[1]], 
                  [0, 0, 1, EXT[2]]], dtype=np.float32)

cat_info = []
max_id = 0
for i, cat in enumerate(cats):
  cat_info.append({'name': cat, 'id': i + 1})

for SPLIT in SPLITS:
  image_set_path = DATA_PATH + 'ImageSets_{}/'.format(SPLIT)
  ann_dir = DATA_PATH + 'training/label_2/'
  calib_dir = DATA_PATH + '{}/calib/'
  splits = ['train', 'val']
  # splits = ['trainval', 'test']
  calib_type = {'train': 'training', 'val': 'training', 'trainval': 'training',
                'test': 'testing'}

  for split in splits:
    ret = {'images': [], 'annotations': [], "categories": cat_info}
    image_set = open(image_set_path + '{}.txt'.format(split), 'r')
    image_to_id = {}
    for line in tqdm(image_set):
      if line[-1] == '\n':
        line = line[:-1]
      image_id = int(line)
      calib_path = calib_dir.format(calib_type[split]) + '{}.txt'.format(line)
      calib = read_clib(calib_path)
      image_info = {'file_name': '{}.jpg'.format(line),
                    'id': int(image_id),
                    'calib': calib.tolist()}
      ret['images'].append(image_info)
      if split == 'test':
        continue
      ann_path = ann_dir + '{}.txt'.format(line)
      # if split == 'val':
      #   os.system('cp {} {}/'.format(ann_path, VAL_PATH))
      # anns = open(ann_path, 'r')

      header = ['class','trunc','occlusion','alphax','xmin','ymin','xmax','ymax','h','w','l','x','y','z','rx','ry','rz','alphay','cx','cy','id']
      annotation = pd.read_csv(ann_path,sep = ' ',names=header)
      
      if DEBUG:
        print(DATA_PATH + 'images/trainval/' + image_info['file_name'])
        image = cv2.imread(
          DATA_PATH + 'images/trainval/' + image_info['file_name'])

      # print(image_info['file_name'])
      for i in range(len(annotation)):
        label = annotation.iloc[i]
        cat_id = cat_ids[label['class']]
        truncated = label['trunc']
        occluded = label['occlusion']
        alphax = float(label['alphax'])
        alphay = float(label['alphay'])
        bbox = [float(label['xmin']),float(label['ymin']),float(label['xmax']),float(label['ymax'])]
        dim = [float(label['h']),float(label['w']),float(label['l'])]
        location = [float(label['x']),float(label['y']),float(label['z'])]
        rotation_x = float(label['rx'])
        rotation_y = float(label['ry'])

        cx = int(label['cx'])
        cy = int(label['cy'])

        fish_id = int(label['id'])

        if fish_id>max_id:
          max_id = fish_id
        # print('dim',dim,'loc',location,'rx',rotation_x,'ry',rotation_y)
        ann = {'image_id': image_id,
               'id': int(len(ret['annotations']) + 1),
               'category_id': cat_id,
               'dim': dim,
               'bbox': _bbox_to_coco_bbox(bbox),
               'depth': location[2],
               'alphax': alphax,
               'alphay': alphay,
               'truncated': truncated,
               'occluded': occluded,
               'location': location,
               'rotation_x': rotation_x,
               'rotation_y': rotation_y,
               'cx':cx,
               'cy':cy,
               'fish_id':fish_id}

        ret['annotations'].append(ann)
        if DEBUG and cat_id != 'DontCare':
          # print(dim, location, rotation_y)
          box_3d = compute_box_3d(dim, location, rotation_y,rotation_x=rotation_x)
          box_2d = project_to_image(box_3d, calib)
          # print('box_2d', box_2d)
          # image = draw_box_3d(image, box_2d)
          image = plot_img(image,box_2d)
          # x = (bbox[0] + bbox[2]) / 2
          # '''
          # print('rot_y, alpha2rot_y, dlt', tmp[0], 
          #       rotation_y, alpha2rot_y(alpha, x, calib[0, 2], calib[0, 0]),
          #       np.cos(
          #         rotation_y - alpha2rot_y(alpha, x, calib[0, 2], calib[0, 0])))
          # '''
          # depth = np.array([location[2]], dtype=np.float32)
          # pt_2d = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
          #                   dtype=np.float32)
          # pt_3d = unproject_2d_to_3d(pt_2d, depth, calib)
          # pt_3d[1] += dim[0] / 2
          # print('pt_3d', pt_3d)
          # print('location', location)
      if DEBUG:
        cv2.imshow('image', image)
        cv2.waitKey()


    print("# images: ", len(ret['images']))
    print("# annotations: ", len(ret['annotations']))
    print('max_id :',max_id)
    # import pdb; pdb.set_trace()
    out_path = '{}/annotations/kitti_{}_{}.json'.format(DATA_PATH, SPLIT, split)
    json.dump(ret, open(out_path, 'w'),cls=NpEncoder)
  
