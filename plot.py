import cv2

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
        
    text_color = (0, 0, 0) # Assuming 'c' is for text color; adjust as needed
    
        # Draw the rectangle
    cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), color, 4)
        # Calculate text size to center it
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        # Calculate center position of text
    text_x = int(x + text_size[0]//2)
    text_y = int(y + text_size[1]//2)
        # Put the text
    cv2.putText(frame, f"{text}, {round(score,2)}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
    
    return frame