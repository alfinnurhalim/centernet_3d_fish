from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import os
import cv2 

def convert_video(input_path, output_folder, max_images):
    # Remove the output folder if it exists
    if os.path.exists(output_folder):
        os.system(f"rm -r {output_folder}")
    
    # Create the output folder
    os.makedirs(output_folder)

    # Open the input video
    cap = cv2.VideoCapture(input_path)

    # Get the video's properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = 512
    height = 512
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine the number of frames to process
    num_frames_to_process = min(frame_count, max_images)

    for frame_number in range(num_frames_to_process):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize the frame to 512x512
        resized_frame = cv2.resize(frame, (width, height))
        
        # Save the resized frame in the output folder
        output_path = os.path.join(output_folder, f"frame_{frame_number:04d}.jpg")
        cv2.imwrite(output_path, resized_frame)

    # Release the video capture object
    cap.release()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count