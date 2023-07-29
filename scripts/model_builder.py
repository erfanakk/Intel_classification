
import torch
import torch.nn as nn
import torch.functional as F

import matplotlib.pyplot as plt
from torchviz import make_dot
from torchsummary import summary





torch.manual_seed(42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")








class model_sigmoid(nn.Module):
    def __init__(self , in_channels=3, num_class=6):
        super(model_sigmoid , self).__init__() 

        self.seq1 = nn.Sequential(
                                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                                nn.Sigmoid(),
                                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                                nn.Sigmoid(),
                                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                nn.Sigmoid(),
                                nn.MaxPool2d(kernel_size=2)
                            )

        self.seq2 = nn.Sequential(
                                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                nn.Sigmoid(),
                                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                nn.Sigmoid(),
                                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                nn.Sigmoid(),
                                nn.MaxPool2d(kernel_size=2)
                            )  

        self.seq3 = nn.Sequential(
                                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                                nn.Sigmoid(),
                                nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                                nn.Sigmoid(),
                                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                                nn.Sigmoid(),
                                nn.MaxPool2d(kernel_size=2),
                            )

        self.fc1 = nn.Linear(1024 * (64 // 8) * (64 // 8), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256 , num_class)
        self.sigmoid = nn.Sigmoid()


        
        self.initialize_weights()

    def forward(self, x):
        x = self.seq1(x)
        x = self.seq2(x)
        x = self.seq3(x)
        x = x.view(x.size(0), -1)
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        out = self.fc3(x)
        return out   

    def initialize_weights(self):
        with torch.no_grad():     
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)










class model_tanh(nn.Module):
    def __init__(self , in_channels=3, num_class=6):
        super(model_tanh , self).__init__() 

        self.seq1 = nn.Sequential(
                                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                                nn.Tanh(),
                                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                                nn.Tanh(),
                                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                nn.Tanh(),
                                nn.MaxPool2d(kernel_size=2)
                            )

        self.seq2 = nn.Sequential(
                                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                nn.Tanh(),
                                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                nn.Tanh(),
                                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                nn.Tanh(),
                                nn.MaxPool2d(kernel_size=2)
                            )  

        self.seq3 = nn.Sequential(
                                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                                nn.Tanh(),
                                nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                                nn.Tanh(),
                                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                                nn.Tanh(),
                                nn.MaxPool2d(kernel_size=2),
                            )

        self.fc1 = nn.Linear(1024 * (64 // 8) * (64 // 8), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256 , num_class)
        self.tanh = nn.Tanh()


        
        self.initialize_weights()

    def forward(self, x):
        x = self.seq1(x)
        x = self.seq2(x)
        x = self.seq3(x)
        x = x.view(x.size(0), -1)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        out = self.fc3(x)
        return out  

    def initialize_weights(self):
        with torch.no_grad():     
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)






class classification_model_simple(nn.Module):
    def __init__(self , in_channels=3, num_class=6):
        super(classification_model_simple , self).__init__() 

        self.seq1 = nn.Sequential(
                                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2)
                            )

        self.seq2 = nn.Sequential(
                                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2)
                            
                            )


        self.seq3 = nn.Sequential(
                                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                               
                                nn.ReLU(),
                                nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2),
                            )  



        self.fc1 = nn.Linear(256 * (64 // 4) * (64 // 4), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256 , num_class)
        self.relu = nn.ReLU()


        
        self.initialize_weights()

    def forward(self, x):
        x = self.seq1(x)
        x = self.seq2(x)
        x = self.seq3(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.fc3(x)
        return out   

    def initialize_weights(self):
        with torch.no_grad():     
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)




class model_drop_norm(nn.Module):
    def __init__(self , in_channels=3, num_class=6):
        super(model_drop_norm, self).__init__() 

        self.seq1 = nn.Sequential(
                                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(),
                                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(),
                                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2)
                            )

        self.seq2 = nn.Sequential(
                                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2)
                            )  

        self.seq3 = nn.Sequential(
                                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                                nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2),
                            )

        self.fc1 = nn.Linear(1024 * (64 // 8) * (64 // 8), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256 , num_class)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.initialize_weights()



    def forward(self, x):
        x = self.seq1(x)
        x = self.seq2(x)
        x = self.seq3(x)  
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        out = self.fc3(x)
        return out 

    def initialize_weights(self):
        with torch.no_grad():    
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)















if '__main__' == __name__:
    MODELS = {
    'model_Sigmoid': model_sigmoid(),
    'model_tanh': model_tanh(),
    'model_simple_relu': classification_model_simple(),
    'model_simple_drop_norm' :  model_drop_norm(),
    
}
    for name_model, model in MODELS.items():

    
    
    
        img = torch.rand(size=(1 ,3,64,64))
        model = model
        print(model(img).shape)

        summary(model.to(device) ,input_size= (3,64,64))
    # dot = make_dot(y.mean(), params=dict(model.named_parameters()))
    # dot.format = 'png'
    # dot.render('model_graph')
