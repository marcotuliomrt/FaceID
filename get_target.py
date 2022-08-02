import cv2 
PATH_TARGET = "/home/marco/miniconda3/envs/FaceID-env/project_faceid/FaceID"

cap = cv2.VideoCapture(0)

i=0
while cap.isOpened():
    bool, frame = cap.read()
    
    # Cutting down the window size to match the negative dataset
    frame = frame[120:120+250, 200:200+250, :]

    cv2.imshow('Input', frame)

    if cv2.waitKey(1) & 0xFF == ord('p'):   
        cv2.imwrite('./data/targets/target_'+str(i)+'.jpg',frame)
        i+=1

    if cv2.waitKey(1) & 0xFF == ord('k'):
        break


cap.release()
cv2.destroyAllWindows()