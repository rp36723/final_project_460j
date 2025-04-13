import cv2

#below captures frames from camera, should be the main loop function
def capture_frames():
    # Open the default camera (0). You can change the index if you have multiple cameras.
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return
    
    print("Press 'control + c' to exit.")

    while True:
        #check if frame exists and save it in variable
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        #frame variable contains captured frame
        #call function to detect ASL signs + display wherever else needed
        #TODO: add function to detect ASL signs

        #below simply displays frame back to user
        cv2.imshow("Live Camera Feed", frame)
        
        #time interval for capturing frames, right now is 1 frame per second
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break
    
    #release frame and delete active windows to end
    #TODO: have the release and destroy be based on image recognition
    #ie: stop/disconnect is signed therefore camera closes
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_frames()
