import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # (B, 3, 32, 32) -> (B, 32, 32, 32)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # -> (B, 64, 32, 32)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),                              # -> (B, 64, 16, 16)

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # -> (B, 128, 16, 16)
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),                              # -> (B, 128, 8, 8)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),                                 # -> (B, 128*8*8)
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)                            # -> (B, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class CNN_cifar(nn.Module):
    def __init__(self, n_class):
        super(CNN_cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.global_fc1 = nn.Linear(128 * 4 * 4, 128)
        self.global_fc2 = nn.Linear(128, n_class)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool1(x)

        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.pool2(x)

        x = self.bn5(F.relu(self.conv5(x)))
        x = self.bn6(F.relu(self.conv6(x)))
        x = self.pool3(x)

        x = x.view(-1, 128 * 4 * 4)

        x = self.global_fc1(x)
        # x = F.relu(self.global_fc1(x))
        x = self.global_fc2(x)

        # return F.softmax(x, dim=1)
        return x

class CNN_fmnist(nn.Module):
    def __init__(self, n_class):
        super(CNN_fmnist, self).__init__()
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.convlayer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.global_fc1 = nn.Linear(64*6*6, 128)
        self.global_fc2 = nn.Linear(128, n_class)

    def forward(self, x):
        x = self.convlayer1(x)        
        x = self.convlayer2(x)
        x = x.view(-1, 64*6*6)       
        x = self.global_fc1(x)
        x = self.global_fc2(x)
        # x = self.global_fc3(x)
        return x

def str2model_fn(name='cifar10'):
    if name == 'cifar10':
        return CNN_cifar
    elif name == 'fashion':
        return CNN_fmnist
    elif name == 'cifar100':
        return CNN_cifar