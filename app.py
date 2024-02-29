import sys
sys.path.append("./yolov9")

import cv2
import torch
import numpy as np

from models.common import DetectMultiBackend
from utils.torch_utils import select_device, smart_inference_mode
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                         increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)


weights = "model/best.pt"
model = DetectMultiBackend(weights)

conf_threshold = 0.25
nms_iou_thres = 0.45
max_det = 1000

# Open default webcam
cap = cv2.VideoCapture(0)  # Use '0' for the default webcam

def plot_bounding_box(model, frame, x, y, w, h, score, cls):
    text = model.names[int(cls)]
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
        
    text_color = (255, 255, 255) # Assuming 'c' is for text color; adjust as needed
    
        # Draw the rectangle
    cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 1)
        # Calculate text size to center it
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        # Calculate center position of text
    text_x = int(x + w//2 - text_size[0]//2)
    text_y = int(y + h//2 + text_size[1]//2)
        # Put the text
    cv2.putText(frame, f"{text}, {round(score,2)}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
    
    return frame

while True:

    # Read frames from the webcam
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if no frame is captured
    im = np.transpose(frame, (2,0,1))
    im = torch.from_numpy(im).to(model.device).float()
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    
    # Detect and track objects in the frame
    pred = model(im, augment=False, visualize=False)[0]
    filtered_pred = non_max_suppression(pred, conf_threshold, nms_iou_thres, None, False, max_det=max_det)
    # Plot results on the frame
    for p, c in zip(filtered_pred[0], ["r", "b", "g", "cyan"]):
        x, y, w, h, score, cls = p.detach().cpu().numpy().tolist()
        frame = plot_bounding_box(model, frame, x, y, w, h, score, cls)
        
    # Display the annotated frame
    cv2.imshow('frame', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()