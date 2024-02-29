import sys
sys.path.append("./yolov9")

import cv2
import pims
import torch
import numpy as np
from tqdm import tqdm
from plot import plot_bounding_box
from pims import Frame


from models.common import DetectMultiBackend
from utils.torch_utils import select_device, smart_inference_mode
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                         increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)

weights = "model/best.pt"
model = DetectMultiBackend(weights)

conf_threshold = 0.01
nms_iou_thres = 0.01
max_det = 1000

video_path = 'video.mp4'
frames = pims.Video(video_path)

# Get video properties
frame_height, frame_width = frames.frame_shape[:2]
fps = frames.frame_rate  # This might not always be available depending on the backend
number_of_frames = len(frames)

# Define the codec and initialize the VideoWriter object to write the new video
factor = 1
resize_width = 640*factor
resize_height = 384*factor
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change 'mp4v' to another codec if needed
out = cv2.VideoWriter('video_labelled.mp4', fourcc, fps, (resize_width, resize_height))

for i in tqdm(range(number_of_frames)):
    frame = frames[i]
    frame = cv2.resize(frame, (resize_width, resize_height))
    # Your frame processing here
    # For example: convert frame to BGR for cv2 compatibility if necessary
    if frame.ndim == 3 and frame.shape[2] == 3:  # If frame is color
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Assuming the necessary preprocessing and object detection steps are performed here
    im = np.transpose(frame, (2,0,1))
    im = torch.from_numpy(im).float()
    im /= 255  # Normalize pixel values
    if len(im.shape) == 3:
        im = im.unsqueeze(0)  # Add batch dimension
    
    # Object detection and plotting code
    pred = model(im, augment=False, visualize=False)[0]
    filtered_pred = non_max_suppression(pred, conf_threshold, nms_iou_thres, None, False, max_det=max_det)
    
    plot_frame = pims.Frame(frame)
    for p, c in zip(filtered_pred[0], ["r", "b", "g", "cyan"]):
        x, y, w, h, score, cls = p.detach().cpu().numpy().tolist()
        frame = plot_bounding_box(model, plot_frame, x, y, w, h, score, cls)
        
    cv2.imshow("preview", plot_frame)
    cv2.waitKey(1)
    out.write(plot_frame)

#Tthe video writer object, and close all OpenCV windows
out.release()