import os
import sys
import cv2
import argparse
import numpy as np
import pandas as pd

from ultralytics import YOLO
from tqdm import trange,tqdm
from shutil import rmtree

weight_path = '/home/alfin/Documents/deep_learning/YOLOv8/model/best.pt'
model = YOLO(weight_path)

def save_result(df,img_dirs):
	if os.path.exists(os.path.join(img_dirs,'..','yolo_inference_list')):
		rmtree(os.path.join(img_dirs,'..','yolo_inference_list'))

	os.makedirs(os.path.join(img_dirs,'..','yolo_inference_list'))

	frames_idx = sorted(df['idx'].unique())

	for frame_idx in frames_idx:
		df_idx = df[df['idx']==frame_idx]
		df_idx.to_csv(os.path.join(img_dirs,'..','yolo_inference_list',str(frame_idx).zfill(6)+'.txt'),header=False,sep=' ',index=False)


def forward(img_dirs,max_img=50):
	imgs = sorted(os.listdir(img_dirs))[:max_img]
	results = model.track(source=img_dirs+'/*.jpg', tracker="bytetrack.yaml",verbose=False)

	out = list()
	for idx,img_path in enumerate(tqdm(imgs)):
		img_ori = cv2.imread(os.path.join(img_dirs,img_path))
		img_ori = cv2.resize(img_ori,(512,512))

		for box_idx in range(len(results[idx].boxes)):
			data = []

			try:
				fish_id = int(results[idx].boxes[box_idx].id.numpy().tolist()[0])
			except Exception as e:
				print('frame',idx,'box idx',box_idx,e)
				continue

			fish_conf = results[idx].boxes[box_idx].conf.numpy().tolist()[0]

			bbox = results[idx].boxes[box_idx].xyxy.numpy().astype(int).tolist()[0]
			xmin,ymin,xmax,ymax = bbox
			w = int(abs(xmax-xmin))
			h = int(abs(ymax-ymin))

			cx = int(xmin + w//2)
			cy = int(ymin + h//2)

			data = data + [0,0,0,0,0,0,fish_conf,idx,cx,cy]
			data = data + bbox
			data = data + [fish_id]

			out.append(data)
	df = pd.DataFrame(out,columns=['h','w','l','z','alphax','alphay','conf','idx','cx','cy','xmin','ymin','xmax','ymax','id'])

	return df
