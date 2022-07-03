
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== Model class ===================================================================

class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(in_features=384, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=2)


    def forward_layers(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))

        x = torch.flatten(x, start_dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def forward(self, input1, input2):
        output1 = self.forward_layers(input1)
        output2 = self.forward_layers(input2)

        return output1, output2



NET = SiameseNet()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NET = NET.to(DEVICE)

