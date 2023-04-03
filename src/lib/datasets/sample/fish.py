from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import pycocotools.coco as coco
import numpy as np
import torch
import json
import cv2
import os
import math
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
import pycocotools.coco as coco

from scipy.spatial.transform import Rotation

class FishDataset(data.Dataset):
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def __getitem__(self, index):
    img_id = self.images[index]
    img_info = self.coco.loadImgs(ids=[img_id])[0]
    img_path = os.path.join(self.img_dir, img_info['file_name'])

    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    img_annos = self.coco.loadAnns(ids=ann_ids)
    boxes = [self._coco_box_to_bbox(ann['bbox']) for ann in img_annos]
    
    # ===========shape============
    hm = np.zeros((self.opt.num_classes, self.opt.output_h, self.opt.output_w), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)

    dep = np.zeros((self.max_objs, 1), dtype=np.float32)
    dim = np.zeros((self.max_objs, 3), dtype=np.float32)
    rot = np.zeros((self.max_objs, 4), dtype=np.float32)

    wh = np.zeros((self.max_objs, 2), dtype=np.float32)

    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    # ===========shape============

    # process img
    img = cv2.imread(img_path)
    if img == None:
      print(img_path)

    img = self._preprocess_input(img)
    
    cts = []
    for k in range(len(boxes)):
      ann = img_annos[k]

      bbox = self._preprocess_bbox(boxes[k])

      cls_id = int(self.cat_ids[1])

      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

      # only taking hm where h and w >0
      if h > 0 and w > 0:

        # HEATMAP AND CENTER  
        cx_3d = ann['cx']
        cy_3d = ann['cy']

        cx_2d = bbox[0] + w/2
        cy_2d = bbox[1] + h/2

        ct = np.array(
          [cx_2d,cy_2d], dtype=np.float32)
        ct_int = ct.astype(np.int32)

        cts.append(ct_int)
        radius = gaussian_radius((h, w))
        radius = max(1, int(radius))
        
        draw_msra_gaussian(hm[0], ct, radius)

        angle_max = (2*np.pi)
        alphaX = np.degrees((ann['alphax'] + angle_max) % angle_max)
        alphaY = np.degrees((ann['alphay'] + angle_max) % angle_max)

        angle = Rotation.from_euler('xyz', [alphaX, alphaY, 0], degrees=True)
        angle = angle.as_quat()
        
        dep[k] = ann['depth']
        dim[k] = ann['dim']
        rot[k] = angle
        # rot[k] = [np.sin(alphaX),np.cos(alphaX),np.sin(alphaY),np.cos(alphaY)]

        wh[k] = [w,h]

        ind[k] = ct_int[1] * self.opt.output_w + ct_int[0]
        reg[k] = ct - ct_int

        reg_mask[k] = 1

    ret = {'input': img, 
          'hm': hm,
          'reg' : reg,
          'reg_mask': reg_mask,
          'ind': ind,
          'dep': dep, 
          'dim': dim, 
          'rot':rot,
          'wh' :wh,
          }
        
    return ret

  def _preprocess_input(self,img):
    inp = (img.astype(np.float32) / 255.)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    return inp

  def _preprocess_bbox(self,bbox):
    bbox = np.array(bbox)
    bbox = bbox / self.opt.down_ratio

    return bbox