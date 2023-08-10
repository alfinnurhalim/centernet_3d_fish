import _init_paths

import os
import cv2
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from shutil import rmtree
from munkres import Munkres

from torchvision.ops import box_iou
from scipy.spatial.transform import Rotation
from sklearn.metrics.pairwise import cosine_similarity

from opts import opts
from utils.image import get_iou,get_2d_from_3d,read_calib
from utils.debugger import Debugger
from datasets.dataset_factory import get_dataset
from models.decode import ddd_decode,fish_decode
from detectors.detector_factory import detector_factory

calib_path = '/home/alfin/Documents/deep_learning/production/centernet_3d_fish/data/000000.txt'
P = read_calib(calib_path)

class FishTracker(object):

	def __init__(self):
		super(FishTracker, self).__init__()

		self.m = Munkres()

		self.max_id = 0
		self.states = list()
		self.max_age = 7

	def update(self,tracklets):

		if len(self.states)==0:
			for i in range(len(tracklets)):
				tracklets[i]['fish_id'] = i
				self.max_id = i

			self.states.append(tracklets)
			print([x['fish_id'] for x in self.states[-1]])
			return tracklets

		self.states.append(tracklets)

		s1 = self.states[-1]
		reid_1 = np.array([x['reid'] for x in s1]).squeeze(axis=1)
		bbox_1 = np.array([x['bbox'] for x in s1])

		adj_frames = self.states[-self.max_age:-1]

		for adj_idx in range(len(adj_frames)):
			s0 = self.states[-2+adj_idx]

			untracked_fish_idx = self._get_linked_tracklets(reid_1,bbox_1,s0)
			if len(untracked_fish_idx) > 1:
				reid_1 = reid_1[untracked_fish_idx]
				bbox_1 = bbox_1[untracked_fish_idx]
			else:
				continue

		if len(untracked_fish_idx) > 0 :
			for neg_idx in untracked_fish_idx:
				self.states[-1][neg_idx]['fish_id'] = self.max_id+1
				self.max_id += 1

		print([x['fish_id'] for x in self.states[-1]])
		return self.states[-1]

	def _get_linked_tracklets(self,reid_1,bbox_1,s0):
		reid_0 = np.array([x['reid'] for x in s0]).squeeze(axis=1)
		bbox_0 = np.array([x['bbox'] for x in s0])
		
		pos_idx,neg_idx = get_iou(bbox_1,bbox_0,len(reid_0),iou_thresh=0.1)

		reid_1_pos = reid_1[pos_idx]
		id_maps = self._get_cosine_similarity(reid_1_pos,reid_0)
		# print(pos_idx,neg_idx,[x[0] for x in id_maps])
		for id_map in id_maps:
			s1_idx,s0_idx = id_map

			s0_idx = s0_idx
			s1_idx = pos_idx[s1_idx]

			self.states[-1][s1_idx]['fish_id'] = s0[s0_idx]['fish_id']

		return neg_idx


	def _get_cosine_similarity(self,reid_1,reid_0):
		cos = (1 - cosine_similarity(reid_1,reid_0))
		cos_index = self.m.compute(cos)

		return cos_index

img_dirs = '/home/alfin/Documents/deep_learning/fish_conversion/data/20221121_centernet_rxry/KITTI/detection/training/sample_video/'
img_dirs = os.path.join(img_dirs,'output_3')

MODEL_PATH = '../models/red_orbitting_3cam_output_3_reid.pth'

# tracking_test
down_ratio = 4
tracker = FishTracker()

# Init model
Dataset = get_dataset('fish_sim', 'fish')
opt = opts().init('{} --load_model {}'.format('fish', MODEL_PATH).split(' '))
opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
detector = detector_factory[opt.task](opt)

# Init Folders
if os.path.exists(os.path.join(img_dirs,'..','label_inference')):
	rmtree(os.path.join(img_dirs,'..','label_inference'))

os.makedirs(os.path.join(img_dirs,'..','label_inference'))

imgs = sorted(os.listdir(img_dirs))[:3]
for idx,img_path in enumerate((imgs)):

	print('frame',idx)
	img_ori = cv2.imread(os.path.join(img_dirs,img_path))
	img_ori = cv2.resize(img_ori,(512,512))

	img = img_ori.copy()

	img_input = detector.pre_process(img)
	img_input = img_input.to(detector.opt.device)

	torch.no_grad()
	torch.cuda.synchronize()

	output = detector.model(img_input)[-1]

	output['hm'] = output['hm'].sigmoid_()
	output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
	torch.cuda.synchronize()

	dets = fish_decode(output['hm'],output['reg'],output['dep'],output['dim'],output['rot'],output['reid'],K=detector.opt.K)
	out = list()

	tracklets = []
	for obj_idx,det in enumerate(dets):
		scores = det['conf'][0]
		class_name = int(det['class'])

		cx = int(det['cx'] * down_ratio)
		cy = int(det['cy'] * down_ratio)

		dim = list(det['dim'])
		depth = det['dep'][0]

		rot = list(det['rot'])

		reid = det['reid'].reshape((1,-1))

		bbox = get_2d_from_3d(P,det)

		if scores > 0.3:
			tracklet = {'reid' : reid,
						'bbox' : bbox,
						'idx'	: idx,
						'obj_idx' : obj_idx,
						'fish_id':None}

			tracklets.append(tracklet)

	tracklets = tracker.update(tracklets)
	# try:
	# 	print(sorted([x['fish_id'] for x in tracklets]))
	# except Exception as e:
	# for i in range(len(tracklets)):
	# 	print(i,tracklets[i]['fish_id'])
		# print(e)
	# 		merged = list()
	# 		merged += dim

	# 		merged.append(depth)

	# 		merged += rot

	# 		merged.append(scores)
	# 		merged.append(int(idx))
	# 		merged.append(cx)
	# 		merged.append(cy)

	# 		merged.append(xmin)
	# 		merged.append(ymin)
	# 		merged.append(xmax)
	# 		merged.append(ymax)
	# 		out.append(merged)

	# if idx == 0 :
	# 	s1 = s0

	# reid_0 = np.array([s0[x]['reid'] for x in list(s0.keys())[:]]).squeeze(axis=1)
	# reid_1 = np.array([x['reid'] for x in tracklets]).squeeze(axis=1)

	# bbox_0 = np.array([s0[x]['bbox'] for x in list(s0.keys())[:]])
	# bbox_1 = np.array([x['bbox'] for x in tracklets])

	# box_pos,box_neg = get_iou(bbox_1,bbox_0,len(reid_0),iou_thresh=0.1)

	# reid_1_pos = reid_1[box_pos]
	# reid_1_neg = reid_1[box_neg]

	# cos = (1 - cosine_similarity(reid_1_pos,reid_0))
	# indexes = m.compute(cos)

	# for enum,pos_idx in enumerate(box_pos):
	# 	row, column = indexes[enum]

	# 	fish_id = list(s0.keys())[column]
	# 	out[pos_idx].append(fish_id)
	# 	if idx > 0:
	# 	    s1[fish_id] = { 'reid' : reid_1[pos_idx].reshape((1,-1)),
	# 	                    'bbox' : bbox_1[pos_idx]
	# 	                    }

	# for enum,neg_idx in enumerate(box_neg):
	# 	fish_id = max_id + 1
	# 	max_id = fish_id
	# 	out[neg_idx].append(fish_id)

	# 	if idx > 0:
	# 		s1[fish_id] = { 'reid' : reid_1[neg_idx].reshape((1,-1)),
	# 						'bbox' : bbox_1[neg_idx]
	# 						}
	# states[idx] = s1

	# df = pd.DataFrame(out,columns=['h','w','l','z','alphax','alphay','conf','idx','cx','cy','xmin','ymin','xmax','ymax','id'])
	# df.to_csv(os.path.join(img_dirs,'..','label_inference',os.path.splitext(os.path.basename(img_path))[0]+'.txt'),header=False,sep=' ',index=False)