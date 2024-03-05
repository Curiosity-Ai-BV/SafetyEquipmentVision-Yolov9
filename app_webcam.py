import sys
sys.path.append("./yolov9")

import cv2
import torch
import numpy as np

from models.common import DetectMultiBackend
from utils.torch_utils import select_device, smart_inference_mode
from utils.general import non_max_suppression

weights = "model/best.pt"
model = DetectMultiBackend(weights)

conf_threshold = 0.25
nms_iou_thres = 0.45
max_det = 1000

# Open default webcam
cap = cv2.VideoCapture(0)

def resize_frame(img, size=(640, 384)):  # Adjusted to (Width x Height)
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

def plot_bounding_box(model, frame, x, y, w, h, score, cls):
    text = model.names[int(cls)]
    color = (255, 0, 0) if text == "person" else (0, 255, 0)  # Example simplification
    text_color = (255, 255, 255)  # White for text

    # Ensure score is a Python float for rounding
    score = score.item() if torch.is_tensor(score) else score  # Convert Tensor to float if needed

    # Draw the rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    # Place the text
    cv2.putText(frame, f"{text}: {round(score, 2)}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to the expected input size of the model
    frame_resized = resize_frame(frame, (640, 384))

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

    cv2.imshow('frame', frame_resized)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
