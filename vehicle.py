
import cv2
import numpy as np

# Load the video (you can use 0 for webcam if needed)
cap = cv2.VideoCapture('video.mp4')


min_width_react = 80   
min_height_react = 80   

count_line_position = 550

# Initialize Background Subtractor
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

def center_handle(x,y,w,h):
     x1=int(w/2)
     y1 =int(h/2)
     cx = x+x1
     cy=y+y1
     return cx,cy

detect = []     
offset = 6        #allowable error btw pixel
counter=0

while True:
    # Read the video frame
    ret, frame1 = cap.read()
    
    if not ret:
        break  # If the video ends, stop the loop
    
    # Convert frame to grayscale
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blur = cv2.GaussianBlur(grey, (3, 3), 5)

    # Apply background subtraction
    img_sub = algo.apply(blur)

    # Dilation for better detection
    dilate = cv2.dilate(img_sub, np.ones((5, 5)))
    
    # Closing operation (morphological operation)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame1, (25, count_line_position) , (1200, count_line_position), (255,127,0), 3)


    for (i,c) in enumerate(contours):
     (x,y,w,h) = cv2.boundingRect(c)
     validate_counter = (w>=min_width_react) and (h>=min_height_react)
     if not validate_counter:
          continue

     cv2.rectangle(frame1, (x,y), (x+w, y+h), (0,255,0)  ,2)
     cv2.putText(frame1, "VEHICLE "+str(counter) , (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,244,0), 2)  


     center = center_handle(x,y,w,h)
     detect.append(center)
     cv2.circle(frame1,center,4,(0,0,255),-1)

     for (x,y) in detect:
          if y<(count_line_position + offset) and y>(count_line_position-offset):
              counter += 1
              cv2.line(frame1 , (25,count_line_position) , (1200,count_line_position), (0,127,255), 3)   
              detect.remove((x,y))
              print("Vehicle counter : "+ str(counter) )


    cv2.putText(frame1, "VEHICLE COUNTER : "+str(counter) , (450,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5)  


    # Display processed frames
   #  cv2.imshow('Background Subtracted', dilatada)
    cv2.imshow('Original Video', frame1)

    # Break on pressing 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
