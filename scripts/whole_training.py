
#import the packages

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




MODELS = {
    'model_Sigmoid': model_sigmoid(),
    'model_tanh': model_tanh(),
    'model_simple_relu': classification_model_simple(),
    'model_simple_drop_norm' :  model_drop_norm(),
    
}


NUM_EPOCHS = 1
NUM_CLASS = 6
IN_CHANNELS = 3
LEARNING_RATE = 0.05
FACTOR_LR = 0.2

device = "cuda" if torch.cuda.is_available() else "cpu"


traindata ,testdata = creat_dataset(path_train= r"data\seg_test" , path_test=r"data\seg_test" , imbalance=True )

class_name = traindata.dataset.classes








for name_model, model in MODELS.items():
    print(f"trainin the {name_model} model")
    writer = SummaryWriter(f"runs/{name_model}")

    
    model = model.to(device)
    img_b , target_b = next(iter(traindata))

    writer.add_graph(model , img_b.to(device))

    grid = torchvision.utils.make_grid(img_b)
    writer.add_image('images', grid, 0)

    loss_fun = nn.CrossEntropyLoss()
    opt = optim.RMSprop(model.parameters(), lr=LEARNING_RATE, alpha=0.9)
    Scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt , factor= FACTOR_LR , patience=15 , verbose=False)



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

            loss = loss_fun(pred , target)
            lossBatch += loss

            loss.backward()

            opt.step()

            opt.zero_grad()

            if (index_batch+1) % 50 == 0:
                mean_loss = lossBatch/50
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
        
        print(f" acc for train {ac_train}  for val {ac_val} for test {ac_test}" )
        print(f" loss for train {loss_train}  for val {loss_val} for test {loss_test}" )
        print(f" time for traning {epoch+1} epoch {end_Epoch-start_Epoch:.3f} sec")
        

        save_model(model , opt , epoch , f"weights/{name_model}.pt")

writer.close()


end = time()
print(f"whole time for traning {end-start:.3f} sec")




# epoch = load_model(model , opt , "modelCNN.pt")


# accurancy(model , testdata , device)

