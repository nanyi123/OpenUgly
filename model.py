import torch
import torch.nn as nn
import torch.nn.functional as F

class hybrid(nn.Module):
    def __init__(self, num_classes=10):
        super(hybrid, self).__init__()
        self.re = resnet18()
        self.Go = GoogLeNet()
        self.fc2 = nn.Linear(128, num_classes)
      
    def forward(self,x):
        x = self.re(x) + self.Go(x)
        x = self.fc2(x)
        return x


class GoogLeNet(nn.Module):
    def __init__(self, output1=128, _initialize_weights=False):
        super(GoogLeNet, self).__init__()
        self.conv9=BasicConv2d(3,64,kernel_size=7,stride=2,padding=3)
        self.MaxPool5=nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv10=BasicConv2d(64,192,kernel_size=3,stride=1,padding=1)
        self.MaxPool6=nn.MaxPool2d(kernel_size=3,stride=2)
        self.Inception1 = Inception(192, 64, 96, 128, 16, 32, 32, 32,32)
        self.conv8 = BasicConv2d(112, 192, kernel_size=5, stride=1, padding=2)
        self.bn4 = nn.BatchNorm2d(192)
        self.MaxPool1 = nn.MaxPool2d(kernel_size=5, stride=5)
        self.fc1 = nn.Linear(192*5*5, output1)

        # if init_weights:
        #     self.tes = LeNet()
        #     self.tes._initialize_weights()

    def forward(self,x):
        x = self.conv9(x)
        x = self.MaxPool5(x)
        x = self.conv10(x)
        x = self.MaxPool6(x)
        x = self.Inception1(x)
        x = self.conv8(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.MaxPool1(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return x

class Inception(nn.Module):
    def __init__(self, in_channels1, c1, c2, c3, c4, c5, c6, c7,c9):
        super(Inception,self).__init__()

        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels1, c1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c1),
            BasicConv2d(c1, c5, kernel_size=1, stride=1)
        )
        # self.branch2 = nn.Sequential(
        #     BasicConv2d(in_channels, c2, kernel_size=1, stride=1),
        #     nn.BatchNorm2d(c2),
        #     BasicConv2d(c2, c6, kernel_size=1, stride=1)
        # )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels1, c2, kernel_size=1, stride=1),
            nn.BatchNorm2d(c2),
            BasicConv2d(c2, c6, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(c6),
            BasicConv2d(c6,c9,kernel_size=3, stride=1,padding=1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels1, c3, kernel_size=1, stride=1),
            nn.BatchNorm2d(c3),
            BasicConv2d(c3, c7, kernel_size=3, stride=1, padding=1)
        )
        self.branch4 = BasicConv2d(in_channels1, c4, kernel_size=1, stride=1)
        nn.BatchNorm2d(c4),
    
    def forward(self,x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x




class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks_num, output2=52, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, output2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet18(output2=128, include_top=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], output2=output2, include_top=include_top)

def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)
