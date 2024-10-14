import cv2
from ultralytics import YOLO

import serial.tools.list_ports

ports = serial.tools.list_ports.comports()
serialInst = serial.Serial()

portlist = []

for onePort in ports:
    portlist.append(str(onePort))
    print(str(onePort))

val = input("Select port: COM")

for x in range(0, len(portlist)):
    if portlist[x].startswith("COM" + str(val)):
        portName = "COM" + str(val)
        print(portName)
serialInst.baudrate = 9600
serialInst.port = portName
serialInst.open()


# Load the YOLO8 model
model = YOLO("project_files/2ndbest.pt")

# Open the video file
video_path = "test2.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Get the height of the frame
        frame_height = frame.shape[0]
        half_height = frame_height // 2
        # Run YOLO8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO8 Tracking", annotated_frame)
        
        if results[0].boxes:  # If boxes is not empty, an object was detected
            for box in results[0].boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]

                # Calculate the center of the box
                box_center_y = (y1 + y2) / 2

                # Check if the center of the box is between the half and the bottom of the window
                if half_height <= box_center_y <= frame_height:
                    print("Object detected in the bottom half of the window!")
                    serialInst.write(b'1')  # Send signal to Arduino
                else:
                    print("Object detected, but not in the bottom half.")
                    serialInst.write(b'0')  # Send different signal to Arduino
        
        else:
            print("No objects detected.")
            serialInst.write(b'0')  # Send signal to Arduino


        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break
# Stop the Arduino signal
serialInst.write(b'0')
# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()