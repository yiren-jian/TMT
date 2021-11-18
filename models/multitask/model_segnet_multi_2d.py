# An implementation of multi-task SegNet
import torch.nn.functional as F
import torch.nn as nn
import torch

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SegNet(nn.Module):
    def __init__(self, output_dim=[13,1,3], num_channels=[64,128,256,512,512]):
        super(SegNet, self).__init__()
        self.output_dim = output_dim
        self.num_task = len(output_dim)
        self.relu = nn.ReLU(inplace=True)

        self.encoder_conv11 = conv3x3(in_planes=3, out_planes=num_channels[0])
        self.encoder_bn11 = nn.BatchNorm2d(num_channels[0])
        self.encoder_conv12 = conv3x3(in_planes=num_channels[0], out_planes=num_channels[0])
        self.encoder_bn12 = nn.BatchNorm2d(num_channels[0])

        self.encoder_conv21 = conv3x3(in_planes=num_channels[0], out_planes=num_channels[1])
        self.encoder_bn21 = nn.BatchNorm2d(num_channels[1])
        self.encoder_conv22 = conv3x3(in_planes=num_channels[1], out_planes=num_channels[1])
        self.encoder_bn22 = nn.BatchNorm2d(num_channels[1])

        self.encoder_conv31 = conv3x3(in_planes=num_channels[1], out_planes=num_channels[2])
        self.encoder_bn31 = nn.BatchNorm2d(num_channels[2])
        self.encoder_conv32 = conv3x3(in_planes=num_channels[2], out_planes=num_channels[2])
        self.encoder_bn32 = nn.BatchNorm2d(num_channels[2])
        self.encoder_conv33 = conv3x3(in_planes=num_channels[2], out_planes=num_channels[2])
        self.encoder_bn33 = nn.BatchNorm2d(num_channels[2])

        self.encoder_conv41 = conv3x3(in_planes=num_channels[2], out_planes=num_channels[3])
        self.encoder_bn41 = nn.BatchNorm2d(num_channels[3])
        self.encoder_conv42 = conv3x3(in_planes=num_channels[3], out_planes=num_channels[3])
        self.encoder_bn42 = nn.BatchNorm2d(num_channels[3])
        self.encoder_conv43 = conv3x3(in_planes=num_channels[3], out_planes=num_channels[3])
        self.encoder_bn43 = nn.BatchNorm2d(num_channels[3])

        self.encoder_conv51 = conv3x3(in_planes=num_channels[3], out_planes=num_channels[4])
        self.encoder_bn51 = nn.BatchNorm2d(num_channels[4])
        self.encoder_conv52 = conv3x3(in_planes=num_channels[4], out_planes=num_channels[4])
        self.encoder_bn52 = nn.BatchNorm2d(num_channels[4])
        self.encoder_conv53 = conv3x3(in_planes=num_channels[4], out_planes=num_channels[4])
        self.encoder_bn53 = nn.BatchNorm2d(num_channels[4])

        self.decoder_conv11 = conv3x3(in_planes=num_channels[4], out_planes=num_channels[4])
        self.decoder_bn11 = nn.BatchNorm2d(num_channels[4])
        self.decoder_conv12 = conv3x3(in_planes=num_channels[4], out_planes=num_channels[4])
        self.decoder_bn12 = nn.BatchNorm2d(num_channels[4])
        self.decoder_conv13 = conv3x3(in_planes=num_channels[4], out_planes=num_channels[3])
        self.decoder_bn13 = nn.BatchNorm2d(num_channels[3])

        self.decoder_conv21 = conv3x3(in_planes=num_channels[3], out_planes=num_channels[3])
        self.decoder_bn21 = nn.BatchNorm2d(num_channels[3])
        self.decoder_conv22 = conv3x3(in_planes=num_channels[3], out_planes=num_channels[3])
        self.decoder_bn22 = nn.BatchNorm2d(num_channels[3])
        self.decoder_conv23 = conv3x3(in_planes=num_channels[3], out_planes=num_channels[2])
        self.decoder_bn23 = nn.BatchNorm2d(num_channels[2])

        self.decoder_conv31 = conv3x3(in_planes=num_channels[2], out_planes=num_channels[2])
        self.decoder_bn31 = nn.BatchNorm2d(num_channels[2])
        self.decoder_conv32 = conv3x3(in_planes=num_channels[2], out_planes=num_channels[2])
        self.decoder_bn32 = nn.BatchNorm2d(num_channels[2])
        self.decoder_conv33 = conv3x3(in_planes=num_channels[2], out_planes=num_channels[1])
        self.decoder_bn33 = nn.BatchNorm2d(num_channels[1])

        self.task_2d = nn.ModuleList()
        for i in range(self.num_task):
            layers = []
            layers.append(conv3x3(in_planes=num_channels[1], out_planes=num_channels[1])) # decoder_conv41
            layers.append(nn.BatchNorm2d(num_channels[1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(conv3x3(in_planes=num_channels[1], out_planes=num_channels[0])) # decoder_conv42
            layers.append(nn.BatchNorm2d(num_channels[0]))
            layers.append(nn.ReLU(inplace=True))
            self.task_2d.append(nn.Sequential(*layers))

        self.task_1d = nn.ModuleList()
        for i in range(self.num_task):
            layers = []
            layers.append(conv3x3(in_planes=num_channels[0], out_planes=num_channels[0])) # decoder_conv51
            layers.append(nn.BatchNorm2d(num_channels[0]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(conv3x3(in_planes=num_channels[0], out_planes=num_channels[0])) # decoder_conv52
            layers.append(nn.BatchNorm2d(num_channels[0]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(conv1x1(in_planes=num_channels[0], out_planes=output_dim[i], stride=1))
            self.task_1d.append(nn.Sequential(*layers))

        # learned parameters in "task uncertainty weighting"
        self.logsigma = nn.Parameter(torch.FloatTensor([-0.5] * self.num_task))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # stage 1
        x = self.relu(self.encoder_bn11(self.encoder_conv11(x)))
        x = self.relu(self.encoder_bn12(self.encoder_conv12(x)))
        x, id1 = F.max_pool2d(x,kernel_size=2, stride=2,return_indices=True) # size=/2, channel=64

        # stage 2
        x = self.relu(self.encoder_bn21(self.encoder_conv21(x)))
        x = self.relu(self.encoder_bn22(self.encoder_conv22(x)))
        x, id2 = F.max_pool2d(x,kernel_size=2, stride=2,return_indices=True) # size=/4, channel=128

        # stage 3
        x = self.relu(self.encoder_bn31(self.encoder_conv31(x)))
        x = self.relu(self.encoder_bn32(self.encoder_conv32(x)))
        x = self.relu(self.encoder_bn33(self.encoder_conv33(x)))
        x, id3 = F.max_pool2d(x,kernel_size=2, stride=2,return_indices=True) # size=/8, channel=256

        # stage 4
        x = self.relu(self.encoder_bn41(self.encoder_conv41(x)))
        x = self.relu(self.encoder_bn42(self.encoder_conv42(x)))
        x = self.relu(self.encoder_bn43(self.encoder_conv43(x)))
        x, id4 = F.max_pool2d(x,kernel_size=2, stride=2,return_indices=True) # size=/16, channel=512

        # stage 5
        x = self.relu(self.encoder_bn51(self.encoder_conv51(x)))
        x = self.relu(self.encoder_bn52(self.encoder_conv52(x)))
        x = self.relu(self.encoder_bn53(self.encoder_conv53(x)))
        x, id5 = F.max_pool2d(x,kernel_size=2, stride=2,return_indices=True) # size=/32, channel=512

        # Stage 5d
        x = F.max_unpool2d(x, id5, kernel_size=2, stride=2)
        x = self.relu(self.decoder_bn11(self.decoder_conv11(x)))
        x = self.relu(self.decoder_bn12(self.decoder_conv12(x)))
        x = self.relu(self.decoder_bn13(self.decoder_conv13(x)))

        # Stage 4d
        x = F.max_unpool2d(x, id4, kernel_size=2, stride=2)
        x = self.relu(self.decoder_bn21(self.decoder_conv21(x)))
        x = self.relu(self.decoder_bn22(self.decoder_conv22(x)))
        x = self.relu(self.decoder_bn23(self.decoder_conv23(x)))

        # Stage 3d
        x = F.max_unpool2d(x, id3, kernel_size=2, stride=2)
        x = self.relu(self.decoder_bn31(self.decoder_conv31(x)))
        x = self.relu(self.decoder_bn32(self.decoder_conv32(x)))
        x = self.relu(self.decoder_bn33(self.decoder_conv33(x)))

         # Stage 2d
        x = F.max_unpool2d(x, id2, kernel_size=2, stride=2)

        y = [0] * self.num_task
        for i in range(self.num_task):
            y[i] = self.task_2d[i](x)

         # Stage 1d
        for i in range(self.num_task):
            y[i] = F.max_unpool2d(y[i], id1, kernel_size=2, stride=2)
            y[i] = self.task_1d[i](y[i])

        return [y[i] for i in range(self.num_task)], self.logsigma
