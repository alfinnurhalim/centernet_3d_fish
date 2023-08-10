import os
import pandas as pd

from tqdm import trange,tqdm

header = ['class','trunc','occlusion','alphax','xmin','ymin','xmax','ymax','h','l','w','x','y','z','rx','ry','rz','alphay','cx','cy','id']

def forward(img_dirs,max_img=50):
	data_path = [os.path.join(img_dirs+'_label',x.split('.')[0]+'.txt') for x in sorted(os.listdir(img_dirs))][:max_img]
	concatenated_df = None

	for idx,label_path in enumerate(tqdm(data_path)):
		ann = pd.read_csv(label_path,sep = ' ',names=header)
		ann['idx'] = idx
		ann['conf'] = 1.0
		if concatenated_df is None:
			concatenated_df = ann
		else:
			concatenated_df = pd.concat([concatenated_df,ann])

	df = pd.DataFrame(concatenated_df,columns=['h','l','w','z','alphax','alphay','conf','idx','cx','cy','xmin','ymin','xmax','ymax','id'])

	return df

def main():
	name = 'general_eval'

	base_dir = '/home/alfin/Documents/deep_learning/fish_conversion/data/20221121_centernet_rxry/KITTI/detection/training/sample_video/'
	img_dirs = os.path.join(base_dir,name)

	test = forward(img_dirs).tail()
	print(test)

if __name__ == '__main__':
	main()
