
#import the packages
import torch.nn.functional as F

from data_setup import creat_dataset
import torch
import torch.nn as nn
from model_builder import *
from torch import optim
from utils import load_model , save_model , accurancy , return_accuracy_loss
from tqdm import tqdm
from time import time
import os
from torch.utils.tensorboard import SummaryWriter
import torchvision



#learning based on time 
#dropout  , norm 
#weight init
 


# Creating a `SummaryWriter` object with the log directory "runs/mnist"
writer = SummaryWriter("runs/intel_simple1")


# Assigning hyperparamets

NUM_EPOCHS = 20
NUM_CLASS = 6
IN_CHANNELS = 3
LEARNING_RATE = 0.05

device = "cuda" if torch.cuda.is_available() else "cpu"



#creating dataset
traindata ,testdata = creat_dataset(path_train= r"data\seg_train" , path_test=r"data\seg_test" , imbalance=True )

class_name = traindata.dataset.classes


# Creating an instance of the `CNN` class and moving it to the `device` (GPU if available)
model = model_PReLU().to(device)
name_model = "model_PReLU"

img_b , target_b = next(iter(traindata))

writer.add_graph(model , img_b.to(device))

grid = torchvision.utils.make_grid(img_b)
writer.add_image('images', grid, 0)

loss_fun = nn.CrossEntropyLoss()
opt = optim.RMSprop(model.parameters(), lr=LEARNING_RATE, alpha=0.9)
Scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt , factor= 0.2 , patience=10 , verbose=True)



n_global = 0
n_embedding = 0
start = time()

for epoch in tqdm(range(NUM_EPOCHS)):
    start_Epoch = time()

    
    model.train()
    lossBatch = 0
    for index_batch,(img,target) in enumerate(traindata):
        
        img = img.to(device)
        target = target.to(device)

        pred = model(img)

        loss = loss_fun(pred,target)
        lossBatch += loss

        loss.backward()

        opt.step()

        opt.zero_grad()

        if (index_batch+1) % 200 == 0:
            mean_loss = lossBatch/200
            print(f"Epoch : {epoch+1} | Batch Index:{index_batch+1}| Loss:{mean_loss:.5f}")
            Scheduler.step(mean_loss)
            lossBatch = 0

    end_Epoch = time()
    writer.add_scalar("epochTime" ,round(end_Epoch-start_Epoch , 3)  , epoch )

    ac_train , loss_train = return_accuracy_loss(traindata , model , loss_fun , device)
    ac_test , loss_test = return_accuracy_loss(testdata , model , loss_fun , device)
    
    writer.add_scalars(
        main_tag = "losses",
        tag_scalar_dict = {
            "train": loss_train,
            "test": loss_test,
            
        }, 
        global_step = epoch
    )
    
    writer.add_scalars(
        main_tag = "accurancy",
        tag_scalar_dict = {
            "train": ac_train,
            "test": ac_test,

        }, 
        global_step = epoch
    )
    print(f" time for traning {epoch+1} epoch {end_Epoch-start_Epoch:.3f} sec")

    save_model(model , opt , epoch , f"weights/{name_model}.pt")

writer.close()


end = time()
print(f"whole time for traning {end-start:.3f} sec")




# epoch = load_model(model , opt , "modelCNN.pt")


# accurancy(model , testdata , device)

