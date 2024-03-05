import sys
sys.path.append("./yolov9")

import cv2
import torch
import numpy as np
from tqdm import tqdm

from models.common import DetectMultiBackend
from utils.general import non_max_suppression

weights = "model/best.pt"
model = DetectMultiBackend(weights)

conf_threshold = 0.20
nms_iou_thres = 0.45
max_det = 1000

# Replace 'Topview-workers.m4v' with the path to your video file
cap = cv2.VideoCapture('Workers-safety-equipment.m4v')

# Determine the video's original FPS and size
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create a VideoWriter object to save the output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video11.mp4', fourcc, fps, (1280, 768))

def resize_frame(img, size=(1280, 768)):
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

def plot_bounding_box(model, frame, x, y, w, h, score, cls):
    text = model.names[int(cls)]
    
    # Set colors for different classes
    if text == "person":
        color = (255, 0, 0) # Blue for person
    elif text == "helmet":
        color = (0, 255, 0) # Green for helmet
    elif text == "no-helmet":
        color = (0, 165, 255) # Orange for no-helmet
    elif text == "vest":
        color = (0, 255, 255) # Cyan for vest
    elif text == "no-vest":
        color = (255, 0, 255) # Magenta for no-vest
    else:
        color = (255, 255, 0) # Yellow for unspecified categories

    score = score.item() if torch.is_tensor(score) else score

    # Draw the rectangle with a thinner line width
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)  # Line width set to 1 for thinner boxes

    # Only add text if the detection is not 'person', with thinner text
    if text != "person":
        cv2.putText(frame, f"{text}: {round(score, 2)}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)  # Text thickness set to 1

    return frame

# Initialize tqdm progress bar
progress_bar = tqdm(total=total_frames, desc="Processing Video", unit="frame")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        progress_bar.close()
        break

    frame_resized = resize_frame(frame, (1280, 768))

    im = np.transpose(frame_resized, (2, 0, 1))
    im = torch.from_numpy(im).to(model.device).float() / 255.0
    im = im.unsqueeze(0)  # Add batch dimension

    pred = model(im, augment=False, visualize=False)[0]
    pred = non_max_suppression(pred, conf_threshold, nms_iou_thres, max_det=max_det)

    if len(pred[0]):
        det = pred[0]
        for *xyxy, conf, cls in det:
            x1, y1, x2, y2 = map(int, xyxy)
            frame_resized = plot_bounding_box(model, frame_resized, x1, y1, x2 - x1, y2 - y1, conf, cls)

    out.write(frame_resized)
    progress_bar.update(1)  # Update progress

cap.release()
out.release()
cv2.destroyAllWindows()
