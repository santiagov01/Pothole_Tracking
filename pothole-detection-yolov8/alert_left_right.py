import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("project_files/2ndbest.pt")

# Open the video file
video_path = "test2.mp4"
cap = cv2.VideoCapture(video_path)


# Get the frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Define the codec and create a VideoWriter object to save the output video
output_path = "output_with_annotations_simple.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Get the frame dimensions (height, width)
        frame_height, frame_width = frame.shape[:2]
        two_thirds_height = (2 * frame_height) // 3  # 2/3 of the height
        half_width = frame_width // 2

        # Run YOLOv8 tracking on the frame
        results = model.track(frame, persist=True)
         # Annotate the frame with the detection results
        #annotated_frame = results[0].plot()  # This returns the frame with annotations (boxes, labels, etc.)
        
        annotated_frame = frame 
        # Initialize variables to track if there are objects on the left or right
        object_on_left = False
        object_on_right = False
        alert_width = 150
        alert_height = 100

        # Check if any objects were detected
        if results[0].boxes:  # If boxes is not empty, objects were detected
            for box in results[0].boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]

                # Calculate the center of the box
                box_center_x = (x1 + x2) / 2
                box_center_y = (y1 + y2) / 2

                # Check if the center of the box is in the bottom third of the frame
                if box_center_y >= two_thirds_height:
                    # Proportional alert size based on object height and width
                    object_width = x2 - x1
                    object_height = y2 - y1
                    alert_width = int(object_width * 0.5)  # 50% of object width
                    alert_height = int(object_height * 0.5)  # 50% of object height

                    # Determine if the object is on the left or right
                    if box_center_x <= half_width:
                        object_on_left = True
                    else:
                        object_on_right = True

        # Display a proportional "LEFT" or "RIGHT" box if an object is detected on that side
        if object_on_left:
            # Draw a proportional red rectangle with "LEFT" in the top-left corner
            cv2.rectangle(annotated_frame, (100, 50), (50 + alert_width, 50 + alert_height), (0, 0, 255), -1)  # Red box
            cv2.putText(annotated_frame, "LEFT", (110, 50 ), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if object_on_right:
            # Draw a proportional green rectangle with "RIGHT" in the top-right corner
            cv2.rectangle(annotated_frame, (frame_width - 100, 50), 
                          (frame_width - 50, 50 + alert_height), (0, 255, 0), -1)  # Green box
            cv2.putText(annotated_frame, "RIGHT", (frame_width - 110, 50 ),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        

        # Write the frame with annotations to the output video
        out.write(annotated_frame)

        # Display the frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
out.release() # Release the VideoWriter object
cv2.destroyAllWindows()
