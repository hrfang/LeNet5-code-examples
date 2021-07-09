import torch
from torch import nn


class LeNet5(nn.Module):
    def __init__(self, num_input_channel=1, activation=nn.modules.activation.ReLU, dropout_rate=0):
        super(LeNet5, self).__init__()
        # assume the input tensor of shape (None,num_input_channel,32,32); None means we don't know mini-batch size yet, 1 is for one channel (grayscale)
        # num_input_channel is the number of input channels; 1 for grayscale images, and 3 for RBG color images
        self.conv1 = nn.Conv2d(num_input_channel, 6, 5)
        self.activation1 = activation()
        # after conv1 layer (5x5 convolution kernels, 6 output channels), the feature map is of shape (None,6,28,28)
        self.pool1 = nn.MaxPool2d(2)
        # after max pooling (kernel_size=2, so is stride), the feature map is of shape (None,6,14,14)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.activation2 = activation()
        # after conv1 layer (5x5 convolution kernels, 16 output channels), the feature map is of shape (None,16,10,10)
        self.pool2 = nn.MaxPool2d(2)
        # after max pooling (kernel_size=2, so is stride), the feature map is of shape (None,16,5,5)
        self.fc1 = nn.Linear(400, 120)
        self.activation3 = activation()
        self.dropout1 = nn.Dropout(p=dropout_rate) if dropout_rate>0 else None
        # note that 16*5*5 = 400
        # the feature map of shape (None,16,5,5) is flattend and of shape (None,400), followed by a fully-connected layer without 120 output features
        # after that, the feature map is now of shape (None,120)
        self.fc2 = nn.Linear(120, 84)
        self.activation4 = activation()
        self.dropout2 = nn.Dropout(p=dropout_rate) if dropout_rate>0 else None
        # after fc2 layer, the feature map is of shape (None,84)
        self.fc3 = nn.Linear(84, 10)
        # after fc3 layer, the feature map is of shape (None,10)
        # we don't need softmax here (just raw network output), since it will be taken care of in the routine torch.nn.CrossEntropyLoss
        # for more information, see https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

    def forward(self, x):
        # feed input x into LeNet-5 network (chain the layers) and get output y
        y = self.conv1(x)
        y = self.activation1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.activation2(y)
        y = self.pool2(y)
        y = torch.flatten(y, 1)  # flatten all dimensions except for the very first (batch dimension)
        y = self.fc1(y)
        y = self.activation3(y)
        if self.dropout1 is not None:
            y = self.dropout1(y)
        y = self.fc2(y)
        y = self.activation4(y)
        if self.dropout2 is not None:
            y = self.dropout2(y)
        y = self.fc3(y)
        return y
