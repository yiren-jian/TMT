# An implementation of same architecture of SegNet but Deconv Layers
import torch.nn as nn
import torch

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def deconv3x3(in_planes, out_planes, stride=2):
    """3x3 transpose convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, output_padding=(1,1), bias=False)


class SegNet(nn.Module):
    def __init__(self, num_classes=13, num_channels=[64,128,256,512,512]):
        super(SegNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.encoder_conv11 = conv3x3(in_planes=3, out_planes=num_channels[0])
        self.encoder_bn11 = nn.BatchNorm2d(num_channels[0])
        self.encoder_conv12 = conv3x3(in_planes=num_channels[0], out_planes=num_channels[0])
        self.encoder_bn12 = nn.BatchNorm2d(num_channels[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_conv21 = conv3x3(in_planes=num_channels[0], out_planes=num_channels[1])
        self.encoder_bn21 = nn.BatchNorm2d(num_channels[1])
        self.encoder_conv22 = conv3x3(in_planes=num_channels[1], out_planes=num_channels[1])
        self.encoder_bn22 = nn.BatchNorm2d(num_channels[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_conv31 = conv3x3(in_planes=num_channels[1], out_planes=num_channels[2])
        self.encoder_bn31 = nn.BatchNorm2d(num_channels[2])
        self.encoder_conv32 = conv3x3(in_planes=num_channels[2], out_planes=num_channels[2])
        self.encoder_bn32 = nn.BatchNorm2d(num_channels[2])
        self.encoder_conv33 = conv3x3(in_planes=num_channels[2], out_planes=num_channels[2])
        self.encoder_bn33 = nn.BatchNorm2d(num_channels[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_conv41 = conv3x3(in_planes=num_channels[2], out_planes=num_channels[3])
        self.encoder_bn41 = nn.BatchNorm2d(num_channels[3])
        self.encoder_conv42 = conv3x3(in_planes=num_channels[3], out_planes=num_channels[3])
        self.encoder_bn42 = nn.BatchNorm2d(num_channels[3])
        self.encoder_conv43 = conv3x3(in_planes=num_channels[3], out_planes=num_channels[3])
        self.encoder_bn43 = nn.BatchNorm2d(num_channels[3])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_conv51 = conv3x3(in_planes=num_channels[3], out_planes=num_channels[4])
        self.encoder_bn51 = nn.BatchNorm2d(num_channels[4])
        self.encoder_conv52 = conv3x3(in_planes=num_channels[4], out_planes=num_channels[4])
        self.encoder_bn52 = nn.BatchNorm2d(num_channels[4])
        self.encoder_conv53 = conv3x3(in_planes=num_channels[4], out_planes=num_channels[4])
        self.encoder_bn53 = nn.BatchNorm2d(num_channels[4])
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.decoder_deconv1 = deconv3x3(in_planes=num_channels[4], out_planes=num_channels[4], stride=2)
        self.decoder_bn1 = nn.BatchNorm2d(num_channels[4])
        self.decoder_conv11 = conv3x3(in_planes=num_channels[4], out_planes=num_channels[4])
        self.decoder_bn11 = nn.BatchNorm2d(num_channels[4])
        self.decoder_conv12 = conv3x3(in_planes=num_channels[4], out_planes=num_channels[3])
        self.decoder_bn12 = nn.BatchNorm2d(num_channels[3])

        self.decoder_deconv2 = deconv3x3(in_planes=num_channels[3], out_planes=num_channels[3], stride=2)
        self.decoder_bn2 = nn.BatchNorm2d(num_channels[3])
        self.decoder_conv21 = conv3x3(in_planes=num_channels[3], out_planes=num_channels[3])
        self.decoder_bn21 = nn.BatchNorm2d(num_channels[3])
        self.decoder_conv22 = conv3x3(in_planes=num_channels[3], out_planes=num_channels[2])
        self.decoder_bn22 = nn.BatchNorm2d(num_channels[2])

        self.decoder_deconv3 = deconv3x3(in_planes=num_channels[2], out_planes=num_channels[2], stride=2)
        self.decoder_bn3 = nn.BatchNorm2d(num_channels[2])
        self.decoder_conv31 = conv3x3(in_planes=num_channels[2], out_planes=num_channels[2])
        self.decoder_bn31 = nn.BatchNorm2d(num_channels[2])
        self.decoder_conv32 = conv3x3(in_planes=num_channels[2], out_planes=num_channels[1])
        self.decoder_bn32 = nn.BatchNorm2d(num_channels[1])

        self.decoder_deconv4 = deconv3x3(in_planes=num_channels[1], out_planes=num_channels[1], stride=2)
        self.decoder_bn4 = nn.BatchNorm2d(num_channels[1])
        self.decoder_conv41 = conv3x3(in_planes=num_channels[1], out_planes=num_channels[0])
        self.decoder_bn41 = nn.BatchNorm2d(num_channels[0])

        self.decoder_deconv5 = deconv3x3(in_planes=num_channels[0], out_planes=num_channels[0], stride=2)
        self.decoder_bn5 = nn.BatchNorm2d(num_channels[0])
        self.decoder_conv51 = conv3x3(in_planes=num_channels[0], out_planes=num_channels[0])
        self.decoder_bn51 = nn.BatchNorm2d(num_channels[0])

        self.pred = conv1x1(in_planes=num_channels[0], out_planes=num_classes, stride=1)

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
        x = self.relu(self.encoder_bn11(self.encoder_conv11(x)))
        x = self.relu(self.encoder_bn12(self.encoder_conv12(x)))
        x = self.pool1(x) # size=/2, channel=64
        x = self.relu(self.encoder_bn21(self.encoder_conv21(x)))
        x = self.relu(self.encoder_bn22(self.encoder_conv22(x)))
        x = self.pool2(x) # size=/4, channel=128
        x = self.relu(self.encoder_bn31(self.encoder_conv31(x)))
        x = self.relu(self.encoder_bn32(self.encoder_conv32(x)))
        x = self.relu(self.encoder_bn33(self.encoder_conv33(x)))
        x = self.pool3(x) # size=/8, channel=256
        x = self.relu(self.encoder_bn41(self.encoder_conv41(x)))
        x = self.relu(self.encoder_bn42(self.encoder_conv42(x)))
        x = self.relu(self.encoder_bn43(self.encoder_conv43(x)))
        x = self.pool4(x) # size=/16, channel=512
        x = self.relu(self.encoder_bn51(self.encoder_conv51(x)))
        x = self.relu(self.encoder_bn52(self.encoder_conv52(x)))
        x = self.relu(self.encoder_bn53(self.encoder_conv53(x)))
        x = self.pool5(x) # size=/32, channel=512

        x = self.relu(self.decoder_bn1(self.decoder_deconv1(x)))
        x = self.relu(self.decoder_bn11(self.decoder_conv11(x)))
        x = self.relu(self.decoder_bn12(self.decoder_conv12(x)))

        x = self.relu(self.decoder_bn2(self.decoder_deconv2(x)))
        x = self.relu(self.decoder_bn21(self.decoder_conv21(x)))
        x = self.relu(self.decoder_bn22(self.decoder_conv22(x)))

        x = self.relu(self.decoder_bn3(self.decoder_deconv3(x)))
        x = self.relu(self.decoder_bn31(self.decoder_conv31(x)))
        x = self.relu(self.decoder_bn32(self.decoder_conv32(x)))

        x = self.relu(self.decoder_bn4(self.decoder_deconv4(x)))
        x = self.relu(self.decoder_bn41(self.decoder_conv41(x)))

        x = self.relu(self.decoder_bn5(self.decoder_deconv5(x)))
        x = self.relu(self.decoder_bn51(self.decoder_conv51(x)))

        x = self.pred(x)

        return x
