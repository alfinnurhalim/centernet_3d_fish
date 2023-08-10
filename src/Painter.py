import os
import cv2
import math
import random

import numpy as np
import pandas as pd

from tqdm import tqdm,trange
from numpy.linalg import inv
from scipy.spatial.transform import Rotation

font = cv2.FONT_HERSHEY_PLAIN 
font_2 = cv2.FONT_HERSHEY_SIMPLEX

class Label(object):
	"""docstring for Label"""
	def __init__(self):
		self.obj_id = None
		self.color = None
		self.center = None
		self.pts = None
		self.bbox = None

class Painter(object):
	"""docstring for Visualizer"""

	def __init__(self,name='result_inf',path='/home/alfin/Documents/deep_learning/synthetic_dataset/'):
		self.img = None
		self.id_colors = dict() 

		self.img_input_size = (512,512)
		self.P = np.array([[281.31866, 0., 256., 0.], 
                [0., 281.31866, 256., 0.], 
                [0., 0., 1., 0.]], dtype=np.float32)

		fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
		self.writer = cv2.VideoWriter(os.path.join(path,name+'.mp4'), fourcc, 25, self.img_input_size)

		self.label = None
		self.perpendicular = None
		self.angle_ranges = 45

	def set_img(self,img):
		self.img = img

	def set_label(self,label):
		ann = Label()

		ann.raw = label
		ann.obj_id = label['id']

		if ann.obj_id not in self.id_colors.keys():
			color = self.random_vibrant_color()
			self.id_colors[ann.obj_id] = [int(color[0]), int(color[1]), int(color[2])]

		ann.color = self.id_colors[ann.obj_id]
		ann.h = label['h']
		ann.w = label['w']
		ann.l = label['l']

		ann.center = (int(label['cx']),int(label['cy']))

		ann.pts = self.get_3d_box(label,ann.center)
		ann.bbox = [int(label['xmin']),int(label['ymin']),int(label['xmax']),int(label['ymax'])]

		self.label = ann

	def draw_dashed_2d_box(self):
		xmin,ymin,xmax,ymax = self.label.bbox
		colored_id = self.label.color
		self.img = self.draw_dashed_rectangle(self.img, (xmin,ymin), (xmax,ymax), colored_id,1,5)

	def draw_2d_box(self):
		box = self.label.bbox
		colored_id = self.label.color
		self.img = cv2.rectangle(self.img, (box[0],box[1]),(box[2],box[3]), colored_id)

	def draw_3d_box(self):
		pts = self.label.pts
		colored_id = self.label.color

		self.img = self.plot_img(self.img,pts,colored_id)

	def draw_frame_num(self,idx):
		self.img = cv2.putText(self.img, str(idx), (10,450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

	def draw_fish_count(self,count):
		self.img = cv2.putText(self.img, str(count), (10,450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

	def draw_size_label(self):
		xmin,ymin,xmax,ymax = self.label.bbox

		self.img = cv2.rectangle(self.img, (xmin,ymax), (xmax,ymax-5), self.label.color,cv2.FILLED)
		self.img = cv2.putText(self.img,str(np.round(self.label.w*100,1))+ ' cm',(xmin,ymax), font,0.5,(255,255,255), 1, cv2.LINE_AA)
	
	def draw_id_label(self):
		xmin,ymin,xmax,ymax = self.label.bbox

		self.img = cv2.rectangle(self.img, (xmin,ymin), (xmax,ymin-5), self.label.color,cv2.FILLED)
		self.img = cv2.putText(self.img,str(self.label.obj_id),(xmin,ymin), font,0.5,(255,255,255), 1, cv2.LINE_AA)

	def compute_acc(self,gt_label,metric):
		if metric == 'w':
			pred = np.round(self.label.w*100,1)
		if metric == 'h':
			pred = np.round(self.label.h*100,1)
		if metric == 'l':
			pred = np.round(self.label.l*100,1)
		gt = np.round(gt_label[metric]*100,1)

		error = np.round(abs(pred-gt),1)
		acc = 100 - np.round((error/gt)*100,1)

		return error,acc

	def draw_size_acc_label(self,gt_label):
		xmin,ymin,xmax,ymax = self.label.bbox

		self.img = cv2.rectangle(self.img, (xmin,ymax), (xmax,ymax+45), self.label.color,cv2.FILLED)

		error,acc = self.compute_acc(gt_label,'w')
		self.img = cv2.putText(self.img,'w : '+str(error)+ ' cm,'+str(acc)+'%',(xmin,ymax + 15), font,1,(255,255,255), 1, cv2.LINE_AA)
		
		error,acc = self.compute_acc(gt_label,'h')
		self.img = cv2.putText(self.img,'h : '+str(error)+ ' cm,'+str(acc)+'%',(xmin,ymax + 30), font,1,(255,255,255), 1, cv2.LINE_AA)
		
		error,acc = self.compute_acc(gt_label,'l')
		self.img = cv2.putText(self.img,'l : '+str(error)+ ' cm,'+str(acc)+'%',(xmin,ymax + 45), font,1,(255,255,255), 1, cv2.LINE_AA)

	def update(self):
		self.writer.write(self.img)

	def save_video(self):
		self.writer.release()

	def random_vibrant_color(self):
		min_saturation = 150
		max_saturation = 255
		min_value = 150
		max_value = 200

		hue = random.randint(0, 179)

		saturation = random.randint(min_saturation, max_saturation)
		value = random.randint(min_value, max_value)

		hsv_color = np.uint8([[[hue, saturation, value]]])
		bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)

		return (int(bgr_color[0][0][0]), int(bgr_color[0][0][1]), int(bgr_color[0][0][2]))
	
	def get_3d_box(self,label,center):
		self.perpendicular = True

		x,y,z = self.get_xy(label,center,self.P)

		ry = self.get_ry(self.img_input_size[0],label)
		rx = self.get_rx(self.img_input_size[1],label) 

		# ry = ry if ry < np.pi else ry - np.pi
		# rx = rx if rx < np.pi else rx - np.pi
		rz = 0

		# if ((ry<np.radians(90+self.angle_ranges)) and (ry>np.radians(90-self.angle_ranges))) :
		# 	perpendicular = True

		if not ((ry<np.radians(90+self.angle_ranges)) and (ry>np.radians(90-self.angle_ranges))) :
			self.perpendicular = False
		if not (rx<np.radians(self.angle_ranges) or (rx>np.radians(180-self.angle_ranges)) ) :
			self.perpendicular = False

		cor = self.get_corners(label,rx,ry,rz,x=x,y=y,z=z)
		pts = self.project_3d(cor)

		return pts

	def get_ry(self,img_size,t):
		return (self.calc_theta_ray_center(img_size,t['cx']) + t['alphax'])

	def get_rx(self,img_size,t):
		return -(self.calc_theta_ray_center(img_size,t['cy'],is_y=True) + t['alphay']) * -1

	def get_xy(self,ann,center,c=0):

		center = np.array(center).reshape((1,2))
		depth = np.array(ann['z']).reshape(1,1)

		return self.imagetocamera(center,depth)

	def calc_theta_ray_center(self,width, center,is_y=False):
		fovx = 2 * np.arctan(width / (2 * self.P[0][0]))

		center = width - center if is_y else center

		dx = center - (width / 2)

		mult = 1
		if dx < 0:
		    mult = -1
		dx = abs(dx)
		angle = np.arctan( (2*dx*np.tan(fovx/2)) / width )
		angle = angle * mult

		angle = fovx/2 - angle
		return angle

	def imagetocamera(self,points,depth):
		"""
		points: (N, 2), N points on X-Y image plane
		depths: (N,), N depth values for points
		projection: (3, 4), projection matrix
		corners: (N, 3), N points on X(right)-Y(down)-Z(front) camera coordinate
		"""
		assert points.shape[1] == 2, "Shape ({}) not fit".format(points.shape)

		corners = np.hstack([points, np.ones(
		    (points.shape[0], 1))]).dot(inv(self.P[:, 0:3]).T)
		assert np.allclose(corners[:, 2], 1)
		corners *= depth.reshape(-1, 1)

		return list(corners[0])

	def get_corners(self,data,rx=0,ry=0,rz=0,x=None,y=None,z=None):

		x = data['x'] if x == None else x
		y = data['y'] if y == None else y
		z = data['z'] if z == None else z

		#     Rotation matrix
		R = Rotation.from_euler('zxy', [rz,rx,ry], degrees=False).as_matrix()

		l = float(data['l'])
		w = float(data['w'])
		h = float(data['h'])

		x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
		y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]
		z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

		corners = np.dot(R,np.vstack([x_corners, y_corners, z_corners]))

		#     translate from origin 
		corners[0,:] = corners[0,:] + x
		corners[1,:] = corners[1,:] + y
		corners[2,:] = corners[2,:] + z

		return corners

	def project_3d(self,corner):
		conn = np.concatenate((corner.T, np.ones((8, 1))), axis=1)
		corners_img_before = np.matmul(conn, self.P.T)
		corners_img = corners_img_before[:, :2] / corners_img_before[:, 2][:, None]

		return corners_img

	def draw_dashed_rectangle(self,image, start_point, end_point, color, thickness=2, dash_length=2):
		x1, y1 = start_point
		x2, y2 = end_point

		# Create a copy of the image to avoid modifying the original
		output_image = image.copy()

		# Calculate the length of the rectangle's sides
		width = abs(x2 - x1)
		height = abs(y2 - y1)

		# Calculate the number of dashes needed for each side
		num_horizontal_dashes = int(np.ceil(width / dash_length))
		num_vertical_dashes = int(np.ceil(height / dash_length))

		# Calculate the step size for each dash
		x_step = width / num_horizontal_dashes
		y_step = height / num_vertical_dashes

		# Draw the horizontal dashed lines
		for i in range(num_horizontal_dashes):
			x_start = int(x1 + i * x_step)
			x_end = int(min(x1 + (i + 1) * x_step, x2))
			if i % 2 == 0:
				cv2.line(output_image, (x_start, y1), (x_end, y1), color, thickness)

			if i % 2 == 0:
				cv2.line(output_image, (x_start, y2), (x_end, y2), color, thickness)

		# Draw the vertical dashed lines
		for i in range(num_vertical_dashes):
			y_start = int(y1 + i * y_step)
			y_end = int(min(y1 + (i + 1) * y_step, y2))
			if i % 2 == 0:
				cv2.line(output_image, (x1, y_start), (x1, y_end), color, thickness)

			if i % 2 == 0:
				cv2.line(output_image, (x2, y_start), (x2, y_end), color, thickness)

		return output_image

	def plot_img(self,plot,pts,color=(255, 0, 0)):

		#color = random_color()
		color_red = [0,0,255]

		# TOP SIDE EFGH
		plot = self.draw_line(plot,pts[4],pts[5],color=color)
		plot = self.draw_line(plot,pts[5],pts[6],color=color)
		plot = self.draw_line(plot,pts[6],pts[7],color=color)
		plot = self.draw_line(plot,pts[7],pts[4],color=color)

		plot = self.draw_line(plot,pts[0],pts[1],color=color)
		plot = self.draw_line(plot,pts[1],pts[2],color=color)
		plot = self.draw_line(plot,pts[2],pts[3],color=color)
		plot = self.draw_line(plot,pts[3],pts[0],color=color)

		# TIANG AE BF CG DH
		plot = self.draw_line(plot,pts[0],pts[4],color=color)
		plot = self.draw_line(plot,pts[1],pts[5],color=color)
		plot = self.draw_line(plot,pts[2],pts[6],color=color)
		plot = self.draw_line(plot,pts[3],pts[7],color=color)

		return plot

	def draw_line(self,img,ptA,ptB,color=(255, 0, 0)):
		start_point = (int(ptA[0]),int(ptA[1]))
		end_point = (int(ptB[0]),int(ptB[1]))
		thickness = 1

		img = cv2.line(img, start_point, end_point, color, thickness)
		return img