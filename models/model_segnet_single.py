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
    def __init__(self, num_classes=13, num_channels=[64,128,256,512,512]):
        super(SegNet, self).__init__()
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

        self.decoder_conv41 = conv3x3(in_planes=num_channels[1], out_planes=num_channels[1])
        self.decoder_bn41 = nn.BatchNorm2d(num_channels[1])
        self.decoder_conv42 = conv3x3(in_planes=num_channels[1], out_planes=num_channels[0])
        self.decoder_bn42 = nn.BatchNorm2d(num_channels[0])

        self.decoder_conv51 = conv3x3(in_planes=num_channels[0], out_planes=num_channels[0])
        self.decoder_bn51 = nn.BatchNorm2d(num_channels[0])
        self.decoder_conv52 = conv3x3(in_planes=num_channels[0], out_planes=num_channels[0])
        self.decoder_bn52 = nn.BatchNorm2d(num_channels[0])
        self.decoder_pred = conv1x1(in_planes=num_channels[0], out_planes=num_classes, stride=1)


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
        x11 = self.relu(self.encoder_bn11(self.encoder_conv11(x)))
        x12 = self.relu(self.encoder_bn12(self.encoder_conv12(x11)))
        x1p, id1 = F.max_pool2d(x12,kernel_size=2, stride=2,return_indices=True) # size=/2, channel=64

        # stage 2
        x21 = self.relu(self.encoder_bn21(self.encoder_conv21(x1p)))
        x22 = self.relu(self.encoder_bn22(self.encoder_conv22(x21)))
        x2p, id2 = F.max_pool2d(x22,kernel_size=2, stride=2,return_indices=True) # size=/4, channel=128

        # stage 3
        x31 = self.relu(self.encoder_bn31(self.encoder_conv31(x2p)))
        x32 = self.relu(self.encoder_bn32(self.encoder_conv32(x31)))
        x33 = self.relu(self.encoder_bn33(self.encoder_conv33(x32)))
        x3p, id3 = F.max_pool2d(x33,kernel_size=2, stride=2,return_indices=True) # size=/8, channel=256

        # stage 4
        x41 = self.relu(self.encoder_bn41(self.encoder_conv41(x3p)))
        x42 = self.relu(self.encoder_bn42(self.encoder_conv42(x41)))
        x43 = self.relu(self.encoder_bn43(self.encoder_conv43(x42)))
        x4p, id4 = F.max_pool2d(x43,kernel_size=2, stride=2,return_indices=True) # size=/16, channel=512

        # stage 5
        x51 = self.relu(self.encoder_bn51(self.encoder_conv51(x4p)))
        x52 = self.relu(self.encoder_bn52(self.encoder_conv52(x51)))
        x53 = self.relu(self.encoder_bn53(self.encoder_conv53(x52)))
        x5p, id5 = F.max_pool2d(x53,kernel_size=2, stride=2,return_indices=True) # size=/32, channel=512

        # Stage 5d
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2)
        x53d = self.relu(self.decoder_bn11(self.decoder_conv11(x5d)))
        x52d = self.relu(self.decoder_bn12(self.decoder_conv12(x53d)))
        x51d = self.relu(self.decoder_bn13(self.decoder_conv13(x52d)))

        # Stage 4d
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2)
        x43d = self.relu(self.decoder_bn21(self.decoder_conv21(x4d)))
        x42d = self.relu(self.decoder_bn22(self.decoder_conv22(x43d)))
        x41d = self.relu(self.decoder_bn23(self.decoder_conv23(x42d)))

        # Stage 3d
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2)
        x33d = self.relu(self.decoder_bn31(self.decoder_conv31(x3d)))
        x32d = self.relu(self.decoder_bn32(self.decoder_conv32(x33d)))
        x31d = self.relu(self.decoder_bn33(self.decoder_conv33(x32d)))

        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
        x22d = self.relu(self.decoder_bn41(self.decoder_conv41(x2d)))
        x21d = self.relu(self.decoder_bn42(self.decoder_conv42(x22d)))

        # Stage 1d
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)
        x12d = self.relu(self.decoder_bn52(self.decoder_conv51(x1d)))
        x11d = self.relu(self.decoder_bn52(self.decoder_conv52(x12d)))
        y = self.decoder_pred(x11d)

        return y
