## Define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## Last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        ## Shape of a Convolutional Layer
        # K - out_channels : the number of filters in the convolutional layer
        # F - kernel_size
        # S - the stride of the convolution
        # P - the padding
        # W - the width/height (square) of the previous layer
        
        # Since there are F*F*D weights per filter
        # The total number of weights in the convolutional layer is K*F*F*D
        
        # 224 by 224 pixels

        # the output Tensor for one image, will have the dimensions: (1, 220, 220)
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv1_bn = nn.BatchNorm2d(32)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        # 220/2 = 110
        # output size = (W-F)/S +1 = (110-4)/1 + 1 = 107
        # the output Tensor for one image, will have the dimensions: (32, 110, 110)
        self.conv2 = nn.Conv2d(32, 64, 4)
        self.conv2_bn = nn.BatchNorm2d(64)
        # output size = (W-F)/S +1 = (53-3)/1 + 1 = 51
        # the output Tensor for one image, will have the dimensions: (64, 54, 54)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv3_bn = nn.BatchNorm2d(128)
        # output size = (W-F)/S +1 = (25-3)/1 + 1 = 23
        # the output Tensor for one image, will have the dimensions: (128, 26, 26)
        self.conv4 = nn.Conv2d(128, 256, 2)
        self.conv4_bn = nn.BatchNorm2d(256)
        # output size = (W-F)/S +1 = (11-3)/1 + 1 = 9
        # the output Tensor for one image, will have the dimensions: (256, 12, 12)
        self.conv5 = nn.Conv2d(256, 512, 1)
        self.conv5_bn = nn.BatchNorm2d(512)
        
        # output size = (W-F)/S +1 = (4-1)/1 + 1 = 4
        # the output Tensor for one image, will have the dimensions: (512, 6, 6)
        
        # Fully-connected (linear) layers
        self.fc1 = nn.Linear(512*6*6, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 68*2)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.25)
        
        
    def forward(self, x):
        ## Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        # 5 conv/relu + pool layers
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = self.pool(F.relu(self.conv4_bn(self.conv4(x))))
        x = self.pool(F.relu(self.conv5_bn(self.conv5(x))))
        
        # Prep for linear layer / Flatten
        x = x.view(x.size(0), -1)
        
        # linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x