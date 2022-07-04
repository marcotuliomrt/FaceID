## About the project 

#### How the scripts work
Action that happens after you run the scripts
- Train.py:
It performs the training (alters the parameters during the process) and in the end saves the parameters on a file with the last character being an index of saved parameter files, for example, on the repository there is already a saved_params_0, if you perform a training, in the end of the training process it will create a saved_params_1, and so on
- live_detection.py:
It loads the parameters from the saved parameters file, opens a webcam image window and makes the inference on these frames. If a detection is made a string shows on the screen top

#### The siamese network
OBS: A important concept to understand the following explanation is "object class". On this context it means a group of objects which the boundaries are the ones the net was made to differentiate. For example, if the net was designed to differentiate human faces, one object class would be composed by many pictures of faces of the SAME PERSON, and the other object class would be composed by many pictures of faces of DIFFERENT PEOPLE 

The net composed by 2 identical (same structure, parameters, ...) convolutional neural nets that compare the output embeddings between the two nets, and the loss is a "dissimilarity" index that represent how different the outputs are. A large value of dissimilarity means the images are different, a small one means they are similar.

In this case, the process where the net "learns" is called one shot leaning, because you need to provide the net just one sample fot it to work. It works because of how its trained. The siamese neural net is fundamentally trained to make a good numerical representation of the objects class (faces, or signatures...), the embedding, what means representing characteristics that can be used to differentiate an object class from another. This is done by the cyclical process of providing the net with 2 images at the same time, some of the time this pair will be composed by 2 images of the same class, and some of the time by images of different classes, the job of the net is to create embedding outputs that are similar when the images are from the same class and from away when the classes are different, if it gets it right the parameters don't have to get updated (positive feedback -> it learned, don't have to update the parameters), if it gets it wrong the parameters get updated (negative feedback -> it didn't learn well, update the parameter)... and from here it's the basic general process of how a neural net learns by changes its parameters. 

The data used to train and test is composed by 3 datasets, one for anchor images, one for positive images and one for negative images. 
- Anchor: Images of an object from a specific objects class
- Positive: Images from the same objects class as the Anchor datasets
- Negative: Images from different objects class as the Anchor and Positive datasets
But what fed to the net ON TRAINING process is a pair of images and a label, some pairs composed by one anchor and a positive image and some pairs composed by an anchor and negative image, and the labels indicate is the respective pair is composed the images of the same class or different ones


## How to use it
There are modules, like model.py and dataset.py, and scripts, like live_detection.py and train.py. Modules are files just used by the scripts, which in fact are responsible for executing functions with defined purposes.
1. Choose the script for the action you want to execute.
- Train the model: train.py
- Use the model to make live detection on camera: live_detection.py

2. Set some prerequisites: Most of the variables are located on the first block of code
- For training the model
    - Make sure the paths to the data sources are right 
    - If desired, you can change the training parameters like epoch number, learning rate, ...
- For live detection
    - Net parameters: You have to make sure the parameters that the model is gonna load are the correct ones, for example: the file saved_params_0 contains the parameters saved after the model training with the pictures specifically of my (@marcotuliomrt) face, so on live detection it detects my face.
    - Dissimilarity limit: You can change the maximum value that the model accepts to print the message on the screen after recognition
    - Detection message: the string that is gonna be plotted on the screen when detection happens.

## References
- Original paper: https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
- Medium article: https://medium.com/sfu-cspmp/do-more-with-less-data-one-shot-learning-with-siamese-neural-networks-760357a2f5cc

- Nicholas Renotte: https://www.youtube.com/watch?v=sQpPaW17TwU&list=PLgNJO2hghbmhHuhURAGbe6KWpiYZt0AMH&index=13
- Nicholas Renotte git: https://github.com/nicknochnack/FaceRecognition/blob/main/Facial%20Verification%20with%20a%20Siamese%20Network%20-%20Part%207.ipynb

- Datahacker.rs https://www.youtube.com/watch?v=9hLcBgnY7cs 
- Datahacker.rs git: https://github.com/maticvl/dataHacker/blob/master/pyTorch/014_siameseNetwork.ipynb
