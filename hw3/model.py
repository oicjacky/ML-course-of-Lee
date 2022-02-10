import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__() # same as: super().__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


class VGG16(nn.Module):
    def __init__(self,
                 linear_batch_norm:bool = False,
                 linear_dropout:bool = False, linear_dropout_rate:float = 0.5):
        super(VGG16, self).__init__() 
        self.linear_batch_norm = linear_batch_norm
        self.linear_dropout = linear_dropout
        self.linear_dropout_rate = linear_dropout_rate
        # input [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]
            
            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 4096),
            nn.BatchNorm1d(4096) if self.linear_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(self.linear_dropout_rate) if self.linear_dropout else nn.Identity(),

            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024) if self.linear_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(self.linear_dropout_rate) if self.linear_dropout else nn.Identity(),

            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


class VGG16_half(nn.Module):
    def __init__(self,
                 linear_batch_norm:bool = False,
                 linear_dropout:bool = False, linear_dropout_rate:float = 0.5):
        super(VGG16_half, self).__init__() 
        self.linear_batch_norm = linear_batch_norm
        self.linear_dropout = linear_dropout
        self.linear_dropout_rate = linear_dropout_rate
        # input [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]
            
            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.Conv2d(256, 256, 3, 1, 1), # [256, 32, 32]
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.Conv2d(512, 512, 3, 1, 1), # [512, 16, 16]
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            # nn.Conv2d(512, 512, 3, 1, 1), # [512, 16, 16]
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            
            # nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            # nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            # nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512*8*8, 1500),
            nn.BatchNorm1d(1500) if self.linear_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(self.linear_dropout_rate) if self.linear_dropout else nn.Identity(),

            nn.Linear(1500, 512),
            nn.BatchNorm1d(512) if self.linear_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(self.linear_dropout_rate) if self.linear_dropout else nn.Identity(),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


if __name__ == "__main__":
    
    model_zoo = {
        'CNN' : Classifier(),
        'vgg16' : VGG16(True, True, 0.5),
        'vgg16_half' : VGG16_half(True, True, 0.5),
    }
    for name, model in model_zoo.items():
        print(name, 'has', sum(p.numel() for p in model.parameters()), 'parameters')
    # CNN        has 12833803 parameters
    # vgg16      has 52619851 parameters
    # vgg16_half has 52321503 parameters