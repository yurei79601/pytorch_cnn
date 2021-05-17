"""
construct CNN model class for classification task
"""
import torch.nn as nn
import config as cfg


class CNN(nn.Module):
    """Create CNN Model"""
    def __init__(self):
        super(CNN, self).__init__()
        # Convolution 1 , input_shape=(1,28,28)
        self.cnn1 = nn.Conv2d(in_channels=3,
                              out_channels=16,
                              kernel_size=5,
                              stride=1,
                              padding=0) #output_shape=(16,24,24)
        self.relu1 = nn.ReLU() # activation
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) #output_shape=(16,12,12)
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16,
                              out_channels=32,
                              kernel_size=5,
                              stride=1,
                              padding=0) #output_shape=(32,8,8)
        self.relu2 = nn.ReLU() # activation
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) #output_shape=(32,4,4)
        self.final_layer_width = int((cfg.input_shape[-1] - 12) / 4)
        # Fully connected 1 ,#input_shape=(32*4*4)
        self.fc1 = nn.Linear(
            32 * self.final_layer_width * self.final_layer_width,
            len(cfg.y_label_list)
        )

    def forward(self, x):
        """get convolutional ouput from x"""
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        # Max pool 1
        out = self.maxpool1(out)
        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)
        # Max pool 2
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        # Linear function (readout)
        out = self.fc1(out)
        return out
