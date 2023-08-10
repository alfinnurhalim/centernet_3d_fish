import _init_paths

import os
import cv2
import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from shutil import rmtree
from numpy.linalg import inv
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation

from detectors.detector_factory import detector_factory
from datasets.dataset_factory import get_dataset
from models.decode import ddd_decode,fish_decode
from utils.debugger import Debugger
from opts import opts

def get_corners(data,rx=0,ry=0,rz=0,x=None,y=None,z=None):
    
    x = data['x'] if x == None else x
    y = data['y'] if y == None else y
    z = data['z'] if z == None else z

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

def project_3d(P,corner):
    conn = np.concatenate((corner.T, np.ones((8, 1))), axis=1)
    corners_img_before = np.matmul(conn, P.T)
    corners_img = corners_img_before[:, :2] / corners_img_before[:, 2][:, None]
    
    return corners_img

def random_color():
    r = random.randint(0, 255)
    rand_color = (255, 0, 0)
    return rand_color

def plot_img(plot,pts):
    
    color = random_color()
    color_red = [0,0,255]
    # TOP SIDE EFGH
    plot = draw_line(plot,pts[4],pts[5],color=color)
    plot = draw_line(plot,pts[5],pts[6],color=color)
    plot = draw_line(plot,pts[6],pts[7],color=color)
    plot = draw_line(plot,pts[7],pts[4],color=color)

    plot = draw_line(plot,pts[0],pts[1],color=color)
    plot = draw_line(plot,pts[1],pts[2],color=color)
    plot = draw_line(plot,pts[2],pts[3],color=color)
    plot = draw_line(plot,pts[3],pts[0],color=color)

    # TIANG AE BF CG DH
    plot = draw_line(plot,pts[0],pts[4],color=color)
    plot = draw_line(plot,pts[1],pts[5],color=color)
    plot = draw_line(plot,pts[2],pts[6],color=color)
    plot = draw_line(plot,pts[3],pts[7],color=color)

    return plot

def draw_line(img,ptA,ptB,color=(255, 0, 0)):
    start_point = (int(ptA[0]),int(ptA[1]))
    end_point = (int(ptB[0]),int(ptB[1]))
    thickness = 1
    
    img = cv2.line(img, start_point, end_point, color, thickness)
    return img

def draw_rect(img,data,color=(0, 255, 0)):
    x0,y0 = int(data['xmin']),int(data['ymin'])
    x1,y1 = int(data['xmax']),int(data['ymax'])
    
    img = cv2.rectangle(img,(x0,y0),(x1,y1), color, 2)
    return img

def roty(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], 
                     [0, 1, 0], 
                     [-s, 0, c]])

def calc_theta_ray(width, xmin,xmax, proj_matrix,is_y=False):
    fovx = 2 * np.arctan(width / (2 * proj_matrix[0][0]))
    center = xmin + abs(xmax-xmin)/2
    
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

def get_theta(width,xmin,xmax,P,is_y=False):
    center = xmin + abs(xmax-xmin)/2
    F = P[0][0]
    dx = center - (width/2)
    dx = width - dx if is_y else dx
    theta = np.arctan(dx/F)
    
    theta = theta + np.pi/2 if is_y else theta
    return theta

def calc_theta_ray(width, xmin,xmax, proj_matrix,is_y=False):
    
    fovx = 2 * np.arctan(width / (2 * proj_matrix[0][0]))
    center = xmin + abs(xmax-xmin)/2
    
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

def calc_theta_ray_center(width, center, proj_matrix,is_y=False):
    fovx = 2 * np.arctan(width / (2 * proj_matrix[0][0]))
    
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

def get_ry(img_size,t,P):
    return (calc_theta_ray_center(img_size,t['cx'],P) + t['alphax'])

def get_rx(img_size,t,P):
    return -(calc_theta_ray_center(img_size,t['cy'],P,is_y=True) + t['alphay']) * -1

def get_xy(ann,center,P,c=0):
    
    center = np.array(center).reshape((1,2))
    depth = np.array(ann['z']).reshape(1,1)

    return imagetocamera(center,depth,P)

def imagetocamera(points, depth, projection):
    assert points.shape[1] == 2, "Shape ({}) not fit".format(points.shape)

    corners = np.hstack([points, np.ones(
        (points.shape[0], 1))]).dot(inv(projection[:, 0:3]).T)
    assert np.allclose(corners[:, 2], 1)
    corners *= depth.reshape(-1, 1)

    return list(corners[0])


def init_model():
    model_path = '../models/red_orbitting_3cam_output_3.pth'
    task = 'fish'
    dataset_name = 'fish_sim'

    Dataset = get_dataset(dataset_name, task)

    # config init
    opt = opts().init('{} --load_model {}'.format(task, model_path).split(' '))
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)

    detector = detector_factory[opt.task](opt)
    debugger = Debugger(dataset=detector.opt.dataset, ipynb=(detector.opt.debug==3),
                            theme=detector.opt.debugger_theme)

    return detector,debugger

def get_camera():
    P = np.array([[318.98947, 0., 256., 0.], 
                [0., 318.98947, 256., 0.], 
                [0., 0., 1., 0.]], dtype=np.float32)

    return P

def get_2dbox_from_3d(merged,center,P):
    img_input_size = (512,512)

    merged_label = pd.DataFrame([merged],columns=['h','l','w','z','alphax','alphay','conf','idx','cx','cy'])
    label = merged_label.iloc[0]
    
    x,y,z = get_xy(label,center,P)

    ry = get_ry(img_input_size[0],label,P)
    rx = get_rx(img_input_size[1],label,P) 
    rz = 0

    cor = get_corners(label,rx,ry,rz,x=x,y=y,z=z)
    pts = project_3d(P,cor)

    xs = [int(x[0]) for x in pts]
    ys = [int(x[1]) for x in pts]

    xmin = min(xs)
    ymin = min(ys)
    xmax = max(xs)
    ymax = max(ys)

    return [xmin,ymin,xmax,ymax]

def forward(img_dirs,max_img=50):
    imgs = sorted(os.listdir(img_dirs))[:max_img]

    detector,debugger = init_model()
    down_ratio = debugger.down_ratio
    result = list()

    for idx,img_path in enumerate(tqdm(imgs)):
        img_ori = cv2.imread(os.path.join(img_dirs,img_path))
        img_ori = cv2.resize(img_ori,(512,512))

        P = get_camera()
        img = img_ori.copy()

        img_input = detector.pre_process(img)
        img_input = img_input.to(detector.opt.device)

        torch.no_grad()
        torch.cuda.synchronize()

        output = detector.model(img_input)[-1]

        output['hm'] = output['hm'].sigmoid_()
        output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
        torch.cuda.synchronize()

        dets = fish_decode(output['hm'],output['reg'],output['dep'],output['dim'],output['rot'],K=detector.opt.K)

        for det in dets:
            scores = det['conf'][0]
            class_name = int(det['class'])

            cx = int(det['cx'] * down_ratio)
            cy = int(det['cy'] * down_ratio)
            center = (cx,cy)

            dim = list(det['dim'])
            depth = det['dep'][0]
            rot = list(np.radians(det['rot']))

            if scores > 0.1:
                
                merged = list()
                merged += dim
                
                merged.append(depth)
                
                merged += rot
                
                merged.append(scores)
                merged.append(int(idx))
                merged.append(cx)
                merged.append(cy)
                
                merged += get_2dbox_from_3d(merged,center,P)
                merged.append(0)
                result.append(merged)

    # whl swaped
    df = pd.DataFrame(result,columns=['h','l','w','z','alphax','alphay','conf','idx','cx','cy','xmin','ymin','xmax','ymax','id'])
    return df

def save_result(df,img_dirs):
    if os.path.exists(os.path.join(img_dirs,'..','label_inference')):
        rmtree(os.path.join(img_dirs,'..','label_inference'))

    os.makedirs(os.path.join(img_dirs,'..','label_inference'))

    print('Saving result.....')
    for idx in df['idx'].unique():
        local_df = df[df['idx']==idx]
        local_df.to_csv(os.path.join(img_dirs,'..','label_inference',str(idx).zfill(6)+'.txt'),header=False,sep=' ',index=False)

def main():
    img_dirs = '/home/alfin/Documents/deep_learning/fish_conversion/data/20221121_centernet_rxry/KITTI/detection/training/sample_video/'
    img_dirs = os.path.join(img_dirs,'output_3_pad')
    max_img = 50

    df = forward(img_dirs)
    save_result(df,img_dirs)
    print(len(df))

if __name__ == '__main__':
    main()