import os
import cv2

import inference_centernet as CenterNet
import inference_yolo as YOLO

from association import match,associate
from Painter import Painter
from tqdm import tqdm

'''
AVAILABLE DRAWING FUNCTION

3d bbox			--> painter.draw_3d_box()
2d bbox			--> painter.draw_2d_box()
dashed 2d bbox	--> painter.draw_dashed_2d_box()
fish size label --> painter.draw_size_label()
fish count 		--> painter.draw_fish_count(len(data))
frame number	--> painter.draw_frame_num(int(idx))
'''

name = 'testing_1_fish'
max_img = 300

base_dir = '/home/alfin/Documents/deep_learning/fish_conversion/data/20221121_centernet_rxry/KITTI/detection/training/sample_video/'
img_dirs = os.path.join(base_dir,name)
imgs_path = [os.path.join(img_dirs,x) for x in sorted(os.listdir(img_dirs))][:max_img]

painter = Painter(name=name)
painter.angle_ranges = 80

centernet_df = CenterNet.forward(img_dirs,max_img)
YOLO_df = YOLO.forward(img_dirs,max_img)
associated_df = match(centernet_df,YOLO_df)

for idx in sorted(associated_df['idx'].unique()):
	path = imgs_path[int(idx)]
	
	data = associated_df[associated_df['idx']==idx]
	img = cv2.imread(path)
	painter.set_img(img)

	for i in range(len(data)):
		label = data.copy().iloc[i]

		painter.set_label(label)

		# if painter.perpendicular :
		painter.draw_3d_box()
		painter.draw_dashed_2d_box()
		painter.draw_size_label()

	painter.draw_fish_count(len(data))

	painter.update()

painter.save_video()
print('Video Saved Sucessfully!!')

