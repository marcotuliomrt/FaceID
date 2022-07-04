import os
import numpy as np
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms


# ==================== Variables ======================================================================

# path where the anchor pictures will be saved
PATH_ANCHOR = "data/anchor"
# path where the positive images will be saved
PATH_POS = "data/positive"
# path where the negative images will be saved
PATH_NEG = "data/negative"
# image size for stardardization 
IMG_SIZE = 100



BATCH_SIZE = 8





# =================== Dataset class =============================================================
class Datasets(torch.utils.data.Dataset):
    def __init__(self, anchor_path, negative_path, positive_path, transform=None, mode=None):
        super(Datasets).__init__()  

        self.anchor_path = anchor_path
        self.positive_path = positive_path
        self.negative_path = negative_path
        
        self.transform = transform
        self.img0 = []
        self.img1 = []
        self.labels = []
        
        self.neg = os.listdir(negative_path)
        self.pos = os.listdir(positive_path)
        self.anch = os.listdir(anchor_path)


        # ****** how the DATASET should look like ****** 
        # img0 = [anch, anch, anch, anch, anch, anch, anch, anch, anch, anch, anch, anch, ..., anch, anch, anch, anch, anch, anch, anch, anch, anch, anch, anch, anch, ...]
        # img1 = [pos, pos, pos, pos, pos, pos, pos, pos, pos, pos, pos, pos, pos, pos, pos, ..., neg, neg, neg, neg, neg, neg, neg, neg,neg, neg, neg, neg, pneg, neg, neg, ...]
        # labels = [1,   1,   1,   1,   1,  1,   1,   1,   1,   1,  1,   1,   1,   1,   1,   ..., 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, ...]


        # create the list of anchors (img0)
        while len(self.img0) < 2*len(self.neg):
            if (2*len(self.neg) - len(self.img0)) > len(self.anch):
                self.img0.extend(self.anch)
            else:
                self.img0.extend(self.anch[:(2*len(self.neg) - len(self.img0))])
        # adds the positives to the img1 list
        while len(self.neg) > len(self.img1):
            if len(self.neg) - len(self.img1) > len(self.pos):
                self.img1.extend(self.pos)
                # append array of zeros because anch represent the same person as pos   *****the labels are dependent on the loss func later created ********
                self.labels.extend(np.zeros(len(self.pos)))
            else:
                self.img1.extend(self.pos[:len(self.neg) - len(self.img1)])
                self.labels.extend(np.zeros(len(self.pos[:len(self.neg) - len(self.labels)])))

        # adds the negatives to the img1 list
        while len(self.img1) < 2*len(self.neg):
            if (2*len(self.neg) - len(self.img1)) >= len(self.neg):
                self.img1.extend(self.neg)
                # append array of ones because anch represent a different person from neg    *****the labels are dependent on the loss func later created ********
                self.labels.extend(np.ones(len(self.neg)))
            else:
                break
    def split(self, start, end):
        return self.img0[start:end], self.img1[start:end], self.labels[start:end]



    def __len__(self):
        return len(self.img0)



    def __getitem__(self, index):
        # get the images from the anchors (img0)
        img0_item = Image.open(self.anchor_path + '/' + self.img0[index])

        # get the image from the positive part of img1
        if index < len(self.neg):
            img1_item = Image.open(self.positive_path + '/' + self.img1[index])
        else:
            # get the image from the negative part of img1
            img1_item = Image.open(self.negative_path + '/' + self.img1[index])
        label_item = self.labels[index]

        if self.transform is not None:
            img0_item = self.transform(img0_item)
            img1_item = self.transform(img1_item)
        else:
            img0_item = np.array(img0_item)
            img1_item = np.array(img1_item)

        return img0_item, img1_item, label_item

            


# ================== Dataloader ===================================================           
# Setting the transformations that are gonna be done on the DATASET
data_transforms = transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Grayscale(num_output_channels=1),
                                       transforms.Resize(IMG_SIZE),
                                       transforms.RandomCrop(IMG_SIZE)
                                       #transforms.Normalize(mean, std)
                                       ])


dataset = Datasets(PATH_ANCHOR, PATH_NEG, PATH_POS, data_transforms)                                       
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)



middle_idx = int(len(dataset)/2)
middle_idx
# defining dataset sizes
trainset_size = round(0.8*len(dataset))
valset_size = len(dataset) - trainset_size

# Defining dataset partition indexes
valset_idx = [idx1 for idx1 in range(middle_idx-round(valset_size/2), middle_idx)] + [idx2 for idx2 in range(middle_idx, middle_idx+round(valset_size/2))]

trainset_idx = [idx1 for idx1 in range(middle_idx-round(valset_size/2))] + [idx2 for idx2 in range(middle_idx+round(valset_size/2), len(dataset))]


# Creating the train and validation datasets
trainset = torch.utils.data.Subset(dataset, trainset_idx)
valset = torch.utils.data.Subset(dataset, valset_idx)

# Creating the dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True)