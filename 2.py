import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)

while True:
    try:  
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        kernel = np.ones((3,3), np.uint8)
        
        # Define region of interest
        roi = frame[80:420, 80:420]
        
        cv2.rectangle(frame, (80, 80), (420, 420), (0, 255, 0), 0)    
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define range of skin color in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Extract skin color image
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Extrapolate the hand to fill dark spots within
        mask = cv2.dilate(mask, kernel, iterations=4)
        
        # Blur the image
        mask = cv2.GaussianBlur(mask, (5, 5), 100)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        text = "No gesture detected"
        if len(contours) > 0:
            cnt = max(contours, key=lambda x: cv2.contourArea(x))
            epsilon = 0.0005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            hull = cv2.convexHull(cnt)
            areahull = cv2.contourArea(hull)
            areacnt = cv2.contourArea(cnt)
            arearatio = ((areahull - areacnt) / areacnt) * 100
            hull = cv2.convexHull(approx, returnPoints=False)
            defects = cv2.convexityDefects(approx, hull)

            if defects is not None:
                num_defects = 0
                for i in range(defects.shape[0]):  # Counting the number of defects
                    s, e, f, d = defects[i, 0]
                    start = tuple(approx[s][0])
                    end = tuple(approx[e][0])
                    far = tuple(approx[f][0])
                    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                    angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c)) * 57

                    # The angle must be less than 90 degrees to count as a defect
                    if angle <= 90:
                        num_defects += 1

                font = cv2.FONT_HERSHEY_SIMPLEX
                if num_defects == 0:
                    cv2.putText(frame, 'rock', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                elif num_defects == 2:
                    cv2.putText(frame, 'scissor', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                elif num_defects >= 4:
                    cv2.putText(frame, 'paper', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                text = "Not enough points for gesture"

        
            # Show the windows
            cv2.imshow('mask', mask)
            cv2.imshow('frame', frame)
        
    except Exception as e:
        print(f"Error: {e}")
        
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
