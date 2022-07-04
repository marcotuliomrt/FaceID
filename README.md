## About the project 

#### How it works 
Action that happens after you run the scripts
- Train.py:
It performs the training (alters the parameters during the process) and in the end saves the parameters on a file with the last character being a index of saved parameter files, for example, on the repository there is already a saved_param_0, if you perform a training, in the end of the training process it will create a saved_params_1, and so on
- live_detection.py:
It loads the parameters from the saved parameters file, opens a webcam image window and makes the inference on this frames. If a detection if made a string shows on the screen top

#### The siamese network
Is a architecture basically composed by 2 identical (same structure, parameters, ...) convolutional neural nets that compare the output embeddings between the two, and the loss is a "dissimilarity" index that represent how different the outputs are.
The data used to train and test is composed by 3 datasets, one for anchor images, one for positive images and one for negative images. 
- Anchor: Images of the "true" object, the one which you want to compare other to ???????????????
- Positive: Images of the true object too ????????????????
- Negative: Images of objects that you wan to distinguish from the real one. If the idea is to distinguish people, the Negatives would be composed be images of random people ???????????



## How to use it
There are modules, like model.py and dataset.py, and scripts, like live_detection.py and train.py. Modules are files just used by the scripts, which in fact are resposable for executing functions with defined purposes.
1. Choose the script for the action you want to execute.
- Train the model: train.py
- Use the model to make live detection on camera: live_detection.py

2. Set some pre-requisites: Most of the variables are located on the fist block of code
- For training the model
    - Make sure the paths to the data sources are rigth 
    - If disired, you can change the training parameters like epoch number, learning rate, ...
- For live detection
    - Net parameters: You have to make sure the parameters that the model is gonna load are the correct ones, for example: the file saved_params_0 contains the parameters saved after the model traing with the pictures specificaly of my (@marcotuliomrt) face, so on live detection it detects my face.
    - Dissimilarity limit: You can change the maximum value that the model accepts to print the messagem on the screen after recognition
    - Detection message: the string that is gonna be ploted on the screen when detection happens.
   

## References
Original paper: https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
Medium article: https://medium.com/sfu-cspmp/do-more-with-less-data-one-shot-learning-with-siamese-neural-networks-760357a2f5cc

Nicholas Renotte: https://www.youtube.com/watch?v=sQpPaW17TwU&list=PLgNJO2hghbmhHuhURAGbe6KWpiYZt0AMH&index=13
Nicholas Renotte git: https://github.com/nicknochnack/FaceRecognition/blob/main/Facial%20Verification%20with%20a%20Siamese%20Network%20-%20Part%207.ipynb

Datahacker.rs https://www.youtube.com/watch?v=9hLcBgnY7cs 
Datahacker.rs git: https://github.com/maticvl/dataHacker/blob/master/pyTorch/014_siameseNetwork.ipynb
