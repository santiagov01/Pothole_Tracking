import cv2 as cv
import torch
import time
import geocoder
import os
from ultralytics import YOLO

# Load YOLOv8 model (from the .pt file)
model = YOLO('project_files/y8best.pt')

# Get class names from your obj.names file
class_name = []
with open(os.path.join("project_files", 'obj.names'), 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

# Define video source (0 for camera or file name for video)
cap = cv.VideoCapture("test2.mp4") 
width  = cap.get(3)
height = cap.get(4)
result = cv.VideoWriter('result.avi', cv.VideoWriter_fourcc(*'MJPG'), 10, (640, 480))

# Geocoder for getting coordinates of potholes
g = geocoder.ip('me')
result_path = "pothole_coordinates"
starting_time = time.time()
frame_counter = 0
i, b = 0, 0

# Detection loop
while True:
    ret, frame = cap.read()
    frame_counter += 1 
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame)  # YOLOv8 detection
    print(type(results))
    print(results)
    detections = results.xyxy[0].cpu().numpy()  # Extract detection results
    
    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection  # Bounding box and confidence
        label = class_name[int(class_id)]  # Get the label from obj.names
        recarea = (x2 - x1) * (y2 - y1)
        area = width * height
        
        if conf >= 0.7 and (recarea / area) <= 0.1 and y2 < 600:
            cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            cv.putText(frame, f"%{round(conf * 100, 2)} {label}", (int(x1), int(y1) - 10), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

            if i == 0 or (time.time() - b) >= 2:
                cv.imwrite(os.path.join(result_path, f'pothole{i}.jpg'), frame)
                with open(os.path.join(result_path, f'pothole{i}.txt'), 'w') as f:
                    f.write(str(g.latlng))
                b = time.time()
                i += 1
    
    # Display FPS
    endingTime = time.time() - starting_time
    fps = frame_counter / endingTime
    cv.putText(frame, f'FPS: {fps:.2f}', (20, 50), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

    # Show frame and save results
    cv.imshow('frame', frame)
    result.write(frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
result.release()
cv.destroyAllWindows()
