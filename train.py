from model import NET, DEVICE
from param_ops import save_params
from loss_func import ContrastiveLoss
from dataset import trainloader
import os
import torch
import time 

# Defining training variables

EPOCHS = 10
LR_i = 0.0005 # learning rate
ITR = 100  # Number of points (iterations) on the graph


OPTIMIZER = torch.optim.Adam(NET.parameters(), lr=LR_i)
LOSS_FUNC = ContrastiveLoss()
SCHEDULER = torch.optim.lr_scheduler.MultiStepLR(OPTIMIZER, milestones=[], gamma=0.5)






# This code creates the sampling for the trainig and validation graph (only interfere on the graph)

###fix this math later ####
len_trainset = 21174
len_valset = 662

ratio = len_trainset/ len_valset
train_steps = round((len_trainset/ITR)/(10/EPOCHS))
val_steps = round(train_steps/ratio)
train_steps, val_steps 


# ============== Training / val loop ====================================================


# cudnn.benchmark  # optimizes runtime ?????

# loads the model on the DEVICE (default: CPU)
NET = NET.to(DEVICE)

# current_step = 0
train_loss_list = []
val_loss_list = []


i_train_time = time.time()

for epoch in range(EPOCHS):
    print("Epoch {}/{}".format(epoch+1, EPOCHS))
    train_counter = 0
    
    i_epoch_time = time.time()
    
    for imgs1, imgs2, label in trainloader:
        
        # loads the data on the GPU 
        imgs1 = imgs1.to(DEVICE)
        imgs2 = imgs2.to(DEVICE)
        label = label.to(DEVICE)
        
        # se the mode for: training 
        NET.train()
        
        # forward propagation
        out1, out2 = NET(imgs1, imgs2)
        # Calculate the loss
        train_loss = LOSS_FUNC(out1, out2, label)

        # zeros the gradient (necessary because on pytorch they are accumulative)
        OPTIMIZER.zero_grad()
        
        # backpropagation
        train_loss.backward()
        # update of the net weights
        OPTIMIZER.step()
        
        
        
        train_counter += 1
        
        # iterating the loss values for later visualization
        if train_counter % train_steps == 0:
            train_loss_list.append(train_loss.item())
            
            
    # update of the learning rate
    SCHEDULER.step()
    f_epoch_time = time.time()
    print('Epoch training time: {} sec, Train loss: {}, Lr: {}'.format(f_epoch_time - i_epoch_time, round(train_loss.item(), 4), round(SCHEDULER.optimizer.param_groups[0]['lr'], 5)))



    # loop for the vallidation -> commented to speed up the training 


    # with torch.no_grad():

    #     val_counter = 0
    #     for imgs1_val, imgs2_val, label_val in valloader:
        
    #         # loads the data on the GPU 
    #         imgs1_val = imgs1_val.to(DEVICE)
    #         imgs2_val = imgs2_val.to(DEVICE)
    #         label_val = label_val.to(DEVICE)
            
    #         # se the mode for: training 
    #         NET.eval()
            
    #         # forward propagation
    #         out1_val, out2_val = NET(imgs1_val, imgs2_val)
    #         # Calculate the loss
    #         val_loss = LOSS_FUNC(out1_val, out2_val, label_val)
            
    #         val_counter += 1
            
    #         # iterating the loss values for later visualization
    #         if val_counter % val_steps == 0:
    #             val_loss_list.append(val_loss.item())


    #     print('Validation loss: {}, Lr: {}'.format(round(val_loss.item() ,4), round(SCHEDULER.optimizer.param_groups[0]['lr'], 5)))



f_train_time = time.time()
print("\n\n", f'Total training time: {round((f_train_time - i_train_time)/60, 1)} min')



# =================== saves the parameters =======================================
i = 0
while i < 100:
    if os.path.exists("saved_params_{0}".format(i)):
        i+=1
        pass
        
    else:
        save_params(NET, "saved_params_{0}".format(i))
        break
        
