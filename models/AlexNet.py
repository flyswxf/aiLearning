import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Conv2d(1,96,11,stride=4,padding=1),nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(96,256,5,stride=1,padding=2),nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(256,384,3,stride=1,padding=1),nn.ReLU(),
            nn.Conv2d(384,384,3,stride=1,padding=1),nn.ReLU(),
            nn.Conv2d(384,256,3,stride=1,padding=1),nn.ReLU(),
            nn.MaxPool2d(3,2),
            
            nn.Flatten(),
            nn.Linear(6400,4096),nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,10),
        )
    
    def forward(self,x):
        return self.layers(x)
