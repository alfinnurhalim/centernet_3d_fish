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
			return tracklets

		self.states.append(tracklets)

		s1 = self.states[-1]
		s0 = self.states[-2]


		reid_1 = np.array([x['reid'] for x in s1]).squeeze(axis=1)
		bbox_1 = np.array([x['bbox'] for x in s1])

		untracked_fish_idx = self._get_linked_tracklets(reid_1,bbox_1,s0)

		# if len(untracked_fish_idx) > 1:
		# 	reid_1 = reid_1[untracked_fish_idx]
		# 	bbox_1 = bbox_1[untracked_fish_idx]
		# else:
		# 	continue

		if len(untracked_fish_idx) > 0 :
			for neg_idx in untracked_fish_idx:
				self.states[-1][neg_idx]['fish_id'] = self.max_id+1
				self.max_id += 1

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

		neg_idx = [i for i,x in enumerate(self.states[-1]) if x['fish_id']==None]
		print('pos_idx = ',len(pos_idx),'neg_idx = ',len(neg_idx),neg_idx)
		return neg_idx


	def _get_cosine_similarity(self,reid_1,reid_0):
		cos = (1 - cosine_similarity(reid_1,reid_0))
		cos_index = self.m.compute(cos)

		return cos_index

img_dirs = '/home/alfin/Documents/deep_learning/fish_conversion/data/20221121_centernet_rxry/KITTI/detection/training/sample_video/'
img_dirs = os.path.join(img_dirs,'output_3')

MODEL_PATH = '../models/red_orbitting_3cam_output_3_reid_fromscratch.pth'

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

imgs = sorted(os.listdir(img_dirs))[:100]
for idx,img_path in enumerate((imgs)):

	print('\nframe',idx)
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

		rot = list(np.radians(det['rot']))

		reid = det['reid'].reshape((1,-1))

		bbox = get_2d_from_3d(P,det)

		if scores > 0.3:
			tracklet = {'idx'	: idx,
						'obj_idx' : obj_idx,

						'scores' : scores,

						'reid' : reid,
						'bbox' : bbox,
						
						'cx' : cx,
						'cy' : cy,
						'dim': dim,
						'depth' : depth,
						'rot' : rot,

						'fish_id':None}

			tracklets.append(tracklet)

	tracklets = tracker.update(tracklets)
	print([[i,x['fish_id']] for i,x in enumerate(tracker.states[-1])])

	for i in range(len(tracklets)):
		tracklet = tracklets[i]

		merged = list()
		merged += tracklet['dim']

		merged.append(tracklet['depth'])

		merged += tracklet['rot']

		merged.append(tracklet['scores'])
		merged.append(int(idx))
		merged.append(tracklet['cx'])
		merged.append(tracklet['cy'])

		merged += tracklet['bbox']
		merged.append(tracklet['fish_id'])
		out.append(merged)

	df = pd.DataFrame(out,columns=['h','w','l','z','alphax','alphay','conf','idx','cx','cy','xmin','ymin','xmax','ymax','id'])
	df.to_csv(os.path.join(img_dirs,'..','label_inference',os.path.splitext(os.path.basename(img_path))[0]+'.txt'),header=False,sep=' ',index=False)
