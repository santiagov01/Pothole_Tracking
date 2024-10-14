#importing necessary libraries
import cv2 as cv
import time
import geocoder
import os

#reading label name from obj.names file
class_name = []
with open(os.path.join("project_files",'obj.names'), 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]
#Class name => Pothole

#importing model weights and config file
#defining the model parameters
net1 = cv.dnn.readNet('project_files/yolov4_tiny.weights', 'project_files/yolov4_tiny.cfg')
net1.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA) #for GPU 
net1.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16) #for GPU
model1 = cv.dnn_DetectionModel(net1) #model1 is the detection model
model1.setInputParams(size=(640, 480), scale=1/255, swapRB=True) #input parameters for the model
#size is the size of the input image
#scale is the factor by which the image is divided by
#swapRB is the flag which indicates that swap first and last channels in 3-channel image is necessary

#defining the video source (0 for camera or file name for video)
cap = cv.VideoCapture("test2.mp4") 
width  = cap.get(3)
height = cap.get(4)
result = cv.VideoWriter('result.avi', 
                         cv.VideoWriter_fourcc(*'MJPG'),
                         10,(640,480)) #result video writer
#VideoWriter_fourcc is a 4-byte code used to specify the video codec.

#defining parameters for result saving and get coordinates
#defining initial values for some parameters in the script
g = geocoder.ip('me') #geocoder for getting coordinates of the potholes 
result_path = "pothole_coordinates" #path for saving the result
starting_time = time.time()
Conf_threshold = 0.5 #confidence threshold
NMS_threshold = 0.4 #non-maximum suppression threshold
frame_counter = 0
i = 0
b = 0

#detection loop
while True:
    ret, frame = cap.read() #reading the frame
    frame_counter += 1 
    if ret == False: #if there is no frame, 
        break
    #analysis the stream with detection model    
    classes, scores, boxes = model1.detect(frame, Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes): #loop for each detected object 
        label = "pothole"
        x, y, w, h = box #coordinates of the detected object
        recarea = w*h #area of the detected object
        area = width*height #area of the frame
        #drawing detection boxes on frame for detected potholes and saving coordinates txt and photo
        if(len(scores)!=0 and scores[0]>=0.7):
            if((recarea/area)<=0.1 and box[1]<600):
                cv.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 1) #drawing rectangle around the detected object
                #writing label and confidence on frame                
                cv.putText(frame, "%" + str(round(scores[0]*100,2)) + " " + label, (box[0], box[1]-10),cv.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)
            
                if(i==0): #if there is no pothole detected before
                    cv.imwrite(os.path.join(result_path,'pothole'+str(i)+'.jpg'), frame) #saving the frame
                    with open(os.path.join(result_path,'pothole'+str(i)+'.txt'), 'w') as f: 
                        f.write(str(g.latlng)) #writing the coordinates of the pothole
                        i=i+1
                if(i!=0):
                    if((time.time()-b)>=2): #if the time difference is greater than 2 seconds
                        cv.imwrite(os.path.join(result_path,'pothole'+str(i)+'.jpg'), frame) #saving the frame
                        with open(os.path.join(result_path,'pothole'+str(i)+'.txt'), 'w') as f: #writing the coordinates of the pothole
                            f.write(str(g.latlng))
                            b = time.time()
                            i = i+1
    #writing fps on frame
    endingTime = time.time() - starting_time #time difference
    fps = frame_counter/endingTime #fps calculation
    cv.putText(frame, f'FPS: {fps}', (20, 50), #writing fps on frame
               cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2) #font and color of the text
    #showing and saving result
    cv.imshow('frame', frame) #showing the frame
    result.write(frame) #saving the frame
    key = cv.waitKey(1) #waiting for a key press
    if key == ord('q'):
        break
    
#end
cap.release()
result.release()
cv.destroyAllWindows()
