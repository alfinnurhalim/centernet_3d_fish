import os
import cv2
import numpy as np
import pandas as pd

from tqdm import trange,tqdm

from onemetric.cv.utils.iou import box_iou_batch,box_iou
from scipy.signal import savgol_filter,butter,filtfilt
from scipy.stats import median_abs_deviation

inf_header = ['h','l','w','z','alphax','alphay','conf','idx','cx','cy','xmin','ymin','xmax','ymax','id']

def associate(df_center,df_yolo):
	match_result = []
	frame_result = []

	bbox_center = df_center[['xmin','ymin','xmax','ymax']].values
	bbox_yolo = df_yolo[['xmin','ymin','xmax','ymax']].values

	iou = box_iou_batch(boxes_true=bbox_yolo,boxes_detection=bbox_center)

	indices = np.argmax(iou, axis=1)
	indices = [[index,z] for index,z in enumerate(indices)]

	for match in indices:

		idx_0 = match[0]
		idx_1 = match[1]

		res_yolo = df_yolo.iloc[idx_0].values.tolist()
		res_centernet = df_center.iloc[idx_1].values.tolist()

		result = res_centernet[:6]+res_yolo[6:]

		frame_result.append(result)

	df = pd.DataFrame(frame_result,columns=inf_header)
	return df

def match(centernet_df,YOLO_df):
	window_size = 5
	order = 1

	fs = 30.0
	cutoff = 0.5

	outlier_window = 30
	threshold = np.pi/4

	df = None

	for idx in range(len(centernet_df['idx'].unique())):
		centernet_data = centernet_df[centernet_df['idx']==idx]
		yolo_data = YOLO_df[YOLO_df['idx']==idx]

		data = associate(centernet_data,yolo_data)

		if df is None:
			df = data
		else:
			df = pd.concat([df,data])
	
	df_input = df.copy()
	df_output = []

	angle_columns = ['alphax','alphay']
	smooth_columns = ['alphax', 'alphay','w','h','l','z']
	result_columns = [x + '_smooth' for x in smooth_columns]
	properties_columns = ['h','w','l']

	save_columns = [x+'_smooth' if x in smooth_columns else x for x in inf_header ]

	for fish_id in df_input['id'].unique():
		tmp = df_input[df_input['id']==fish_id].sort_values('idx')

		if len(tmp) < 15:
			continue
		for col in smooth_columns:

			tmp[col+'_filtered'] = remove_outliers(tmp[col].values, outlier_window, threshold)
			tmp[col] = butter_lowpass_filter(tmp[col+'_filtered'], cutoff, fs, order)

			if col in properties_columns:
				tmp[col] = tmp[col].median()

		if len(df_output) == 0:
			df_output = tmp
		else:
			df_output =  pd.concat([df_output,tmp])

	df_output = df_output.reset_index(drop=True)
	df_output = df_output[inf_header]
	return df_output

def butter_lowpass_filter(data, cutoff, fs, order):
    data = data.values
    
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def normalize_angle(df,limit = np.pi):
    data = df.values
    data = np.where(data < limit, data ,data - limit)
    
    return data

def remove_outliers(signal_data, window_size, threshold):
    filtered_data = signal_data.copy()
    for i in range(window_size, len(signal_data) - window_size):
        window = signal_data[i - window_size: i + window_size + 1]
        median = np.median(window)
        mad = median_abs_deviation(window)
        if abs(signal_data[i] - median) > threshold * mad:
            filtered_data[i] = median
    
    # Special handling for the last frame
    last_frame = signal_data[-1]
    window = filtered_data[-2*window_size:-window_size]
    median = np.median(window)
    mad = median_abs_deviation(window)
    if abs(last_frame - median) > threshold * mad:
        filtered_data[-1] = median
    
    return filtered_data