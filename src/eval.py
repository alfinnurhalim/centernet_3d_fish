import os
import cv2

import inference_centernet as CenterNet
import inference_yolo as YOLO
import inference_synthetic as Dataset

from association import match,associate
from Painter import Painter
from tqdm import tqdm,trange

'''
AVAILABLE DRAWING FUNCTION

3d bbox         --> painter.draw_3d_box()
2d bbox         --> painter.draw_2d_box()
dashed 2d bbox  --> painter.draw_dashed_2d_box()
fish size label --> painter.draw_size_label()
fish count      --> painter.draw_fish_count(len(data))
frame number    --> painter.draw_frame_num(int(idx))
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

synthetic_df = Dataset.forward(img_dirs,max_img)
eval_df = associate(associated_df,synthetic_df)

data = eval_df

for idx in tqdm(sorted(data['idx'].unique())):
    path = imgs_path[int(idx)]
    
    label = data[data['idx']==idx]
    gt = synthetic_df[synthetic_df['idx']==idx]

    img = cv2.imread(path)
    painter.set_img(img)

    for i in range(len(label)):
        ann = label.copy().iloc[i]

        ann_gt = gt[gt['id'] == ann['id']]
        ann_gt = ann_gt.iloc[0]

        painter.set_label(ann)

        # if painter.perpendicular :
        painter.draw_3d_box()
        painter.draw_dashed_2d_box()
        # painter.draw_size_label()
        painter.draw_size_acc_label(ann_gt)

    painter.draw_fish_count(len(label))

    painter.update()

painter.save_video()
print('Video Saved Sucessfully!!')

