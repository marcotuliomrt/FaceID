

from dataset import IMG_SIZE, transforms
from param_ops import load_params
from model import NET, DEVICE
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch
import cv2


# --------------- variables ---------------------------------------------------------------------------------
PATH_TARGET = "data/targets/target.jpg"
max_dist = 0.9 # Maximum euclidean distance that is gonna be accepted to plot the text on the screen 
text = "<MARCO>" # text is gonna be ploted


# ------------------------------------------------------------------------------------------------------------


# Loads the saves parameters from the defined path
load_params(NET, "saved_params_0")

# formating function used on the real-time test -> both images entering the model must have the exact same transformations (resize, stardization, etc)
def img_format(array):
    resized = cv2.resize(array, (IMG_SIZE, IMG_SIZE), interpolation= cv2.INTER_LINEAR)
    grayed = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    transf = transforms.ToTensor()(grayed).view(100, 100)
    return transf


# Selecting a image that is gonna be used to check similarity with the live capture
PIL_img = Image.open(PATH_TARGET) # PIL
unform_check_img = np.array(PIL_img) # raw array
check_img = img_format(unform_check_img) # formated array
# set the right format to enter the net (a 4 dimensional tensor)
check_img = check_img.view(1, 1, 100, 100)

plt.imshow(check_img.view(100, 100))




# Loop for feeding the model with frames from a camera, displying its euclidean distance with a chosen image (target) and identifing 

cap = cv2.VideoCapture(0)

while cap.isOpened():
    bool, frame = cap.read()
    
    # Cutting down the window size to match the negative dataset
    frame = frame[120:120+250, 200:200+250, :]
    

    NET.eval()
    with torch.no_grad():
        # makes a batch of copies of the same captured frame

        # Formating the image before pass it to the model -> for traiing: done it on the dataloaders 
        image3  = img_format(frame)

        target_img = torch.tensor(np.array([image3.numpy()]))
        # set the right format to enter the net (a 4 dimensional tensor)
        target_img = target_img.view(1, 1, 100, 100)

        # Send the data to the device
        target_img = target_img.float().to(DEVICE)
        check_img = check_img.float().to(DEVICE)
        NET = NET.to(DEVICE)

    
        # Passes the data to the model
        output1, output2 = NET(check_img, target_img)
        # Calculates the "distance" between the tokens
        euclidean_distance = F.pairwise_distance(output1.detach(), output2.detach()).item()
        concat = torch.cat((check_img, target_img), 0)

        # displays the euclidean distance on the screen
        cv2.putText(frame, f"Diss: {euclidean_distance:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        if euclidean_distance < max_dist:
            cv2.putText(frame, text, (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


        cv2.imshow('Input', frame)


    if cv2.waitKey(1) & 0xFF == ord('k'):
        break


cap.release()
cv2.destroyAllWindows()