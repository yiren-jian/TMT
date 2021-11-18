# An implementation of same architecture of SegNet but Deconv Layers
import torch.nn as nn
import torch

def conv7x7(in_planes, out_planes, stride=2, groups=1):
    """7x7 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=(3,3), groups=groups, bias=False)

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

def deconv1x1(in_planes, out_planes, stride=2):
    """1x1 transpose convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, output_padding=(1,1), bias=False)

class SegNet(nn.Module):
    def __init__(self, output_dim=[13,1,3], num_channels=[32,64,128,256]):
        super(SegNet, self).__init__()
        self.output_dim = output_dim
        self.num_task = len(output_dim)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = conv7x7(in_planes=3, out_planes=num_channels[0], stride=2)
        self.bn1 = nn.BatchNorm2d(num_channels[0])
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Encoder stage 1
        self.encoder_stage1_conv1 = conv3x3(in_planes=num_channels[0], out_planes=num_channels[0])
        self.encoder_stage1_bn1 = nn.BatchNorm2d(num_channels[0])
        self.encoder_stage1_conv2 = conv3x3(in_planes=num_channels[0], out_planes=num_channels[0])
        self.encoder_stage1_bn2 = nn.BatchNorm2d(num_channels[0])
        self.encoder_stage1_conv3 = conv3x3(in_planes=num_channels[0], out_planes=num_channels[0])
        self.encoder_stage1_bn3 = nn.BatchNorm2d(num_channels[0])
        self.encoder_stage1_conv4 = conv3x3(in_planes=num_channels[0], out_planes=num_channels[0])
        self.encoder_stage1_bn4 = nn.BatchNorm2d(num_channels[0])

        self.downsample_conv12 = conv1x1(in_planes=num_channels[0], out_planes=num_channels[1], stride=2)
        self.downsample_bn12 = nn.BatchNorm2d(num_channels[1])
        # Encoder stage 2
        self.encoder_stage2_conv1 = conv3x3(in_planes=num_channels[0], out_planes=num_channels[1], stride=2)
        self.encoder_stage2_bn1 = nn.BatchNorm2d(num_channels[1])
        self.encoder_stage2_conv2 = conv3x3(in_planes=num_channels[1], out_planes=num_channels[1])
        self.encoder_stage2_bn2 = nn.BatchNorm2d(num_channels[1])
        self.encoder_stage2_conv3 = conv3x3(in_planes=num_channels[1], out_planes=num_channels[1])
        self.encoder_stage2_bn3 = nn.BatchNorm2d(num_channels[1])
        self.encoder_stage2_conv4 = conv3x3(in_planes=num_channels[1], out_planes=num_channels[1])
        self.encoder_stage2_bn4 = nn.BatchNorm2d(num_channels[1])

        self.downsample_conv23 = conv1x1(in_planes=num_channels[1], out_planes=num_channels[2], stride=2)
        self.downsample_bn23 = nn.BatchNorm2d(num_channels[2])
        # Encoder stage 3
        self.encoder_stage3_conv1 = conv3x3(in_planes=num_channels[1], out_planes=num_channels[2], stride=2)
        self.encoder_stage3_bn1 = nn.BatchNorm2d(num_channels[2])
        self.encoder_stage3_conv2 = conv3x3(in_planes=num_channels[2], out_planes=num_channels[2])
        self.encoder_stage3_bn2 = nn.BatchNorm2d(num_channels[2])
        self.encoder_stage3_conv3 = conv3x3(in_planes=num_channels[2], out_planes=num_channels[2])
        self.encoder_stage3_bn3 = nn.BatchNorm2d(num_channels[2])
        self.encoder_stage3_conv4 = conv3x3(in_planes=num_channels[2], out_planes=num_channels[2])
        self.encoder_stage3_bn4 = nn.BatchNorm2d(num_channels[2])

        self.downsample_conv34 = conv1x1(in_planes=num_channels[2], out_planes=num_channels[3], stride=2)
        self.downsample_bn34 = nn.BatchNorm2d(num_channels[3])
        # Encoder stage 4
        self.encoder_stage4_conv1 = conv3x3(in_planes=num_channels[2], out_planes=num_channels[3], stride=2)
        self.encoder_stage4_bn1 = nn.BatchNorm2d(num_channels[3])
        self.encoder_stage4_conv2 = conv3x3(in_planes=num_channels[3], out_planes=num_channels[3])
        self.encoder_stage4_bn2 = nn.BatchNorm2d(num_channels[3])
        self.encoder_stage4_conv3 = conv3x3(in_planes=num_channels[3], out_planes=num_channels[3])
        self.encoder_stage4_bn3 = nn.BatchNorm2d(num_channels[3])
        self.encoder_stage4_conv4 = conv3x3(in_planes=num_channels[3], out_planes=num_channels[3])
        self.encoder_stage4_bn4 = nn.BatchNorm2d(num_channels[3])

        # Decoder stage 4
        self.decoder_stage4_conv1 = conv3x3(in_planes=num_channels[3], out_planes=num_channels[3])
        self.decoder_stage4_bn1 = nn.BatchNorm2d(num_channels[3])
        self.decoder_stage4_conv2 = conv3x3(in_planes=num_channels[3], out_planes=num_channels[3])
        self.decoder_stage4_bn2 = nn.BatchNorm2d(num_channels[3])

        self.upsample_conv43 = deconv1x1(in_planes=num_channels[3], out_planes=num_channels[2], stride=2)
        self.upsample_bn43 = nn.BatchNorm2d(num_channels[2])

        self.decoder_stage4_conv3 = conv3x3(in_planes=num_channels[3], out_planes=num_channels[3])
        self.decoder_stage4_bn3 = nn.BatchNorm2d(num_channels[3])
        self.decoder_stage4_conv4 = deconv3x3(in_planes=num_channels[3], out_planes=num_channels[2], stride=2)
        self.decoder_stage4_bn4 = nn.BatchNorm2d(num_channels[2])


        # Decoder stage 3
        self.decoder_stage3_conv1 = conv3x3(in_planes=num_channels[2], out_planes=num_channels[2])
        self.decoder_stage3_bn1 = nn.BatchNorm2d(num_channels[2])
        self.decoder_stage3_conv2 = conv3x3(in_planes=num_channels[2], out_planes=num_channels[2])
        self.decoder_stage3_bn2 = nn.BatchNorm2d(num_channels[2])

        self.upsample_conv32 = deconv1x1(in_planes=num_channels[2], out_planes=num_channels[1], stride=2)
        self.upsample_bn32 = nn.BatchNorm2d(num_channels[1])

        self.decoder_stage3_conv3 = conv3x3(in_planes=num_channels[2], out_planes=num_channels[2])
        self.decoder_stage3_bn3 = nn.BatchNorm2d(num_channels[2])
        self.decoder_stage3_conv4 = deconv3x3(in_planes=num_channels[2], out_planes=num_channels[1], stride=2)
        self.decoder_stage3_bn4 = nn.BatchNorm2d(num_channels[1])

        self.task = nn.ModuleList()
        for i in range(self.num_task):
            layers = []
            # Decoder stage 2
            layers.append(conv3x3(in_planes=num_channels[1], out_planes=num_channels[1]))
            layers.append(nn.BatchNorm2d(num_channels[1]))
            layers.append(conv3x3(in_planes=num_channels[1], out_planes=num_channels[1]))
            layers.append(nn.BatchNorm2d(num_channels[1]))

            layers.append(conv3x3(in_planes=num_channels[1], out_planes=num_channels[1]))
            layers.append(nn.BatchNorm2d(num_channels[1]))
            layers.append(deconv3x3(in_planes=num_channels[1], out_planes=num_channels[0], stride=2))
            layers.append(nn.BatchNorm2d(num_channels[0]))

            # Decoder stage 1
            layers.append(conv3x3(in_planes=num_channels[0], out_planes=num_channels[0]))
            layers.append(nn.BatchNorm2d(num_channels[0]))
            layers.append(conv3x3(in_planes=num_channels[0], out_planes=num_channels[0]))
            layers.append(nn.BatchNorm2d(num_channels[0]))
            layers.append(conv3x3(in_planes=num_channels[0], out_planes=num_channels[0]))
            layers.append(nn.BatchNorm2d(num_channels[0]))
            layers.append(conv3x3(in_planes=num_channels[0], out_planes=num_channels[0]))
            layers.append(nn.BatchNorm2d(num_channels[0]))

            layers.append(deconv3x3(in_planes=num_channels[0], out_planes=num_channels[0], stride=2))
            layers.append(nn.BatchNorm2d(num_channels[0]))
            layers.append(conv3x3(in_planes=num_channels[0], out_planes=num_channels[0]))
            layers.append(nn.BatchNorm2d(num_channels[0]))

            layers.append(deconv3x3(in_planes=num_channels[0], out_planes=num_channels[0], stride=2))
            layers.append(nn.BatchNorm2d(num_channels[0]))
            layers.append(conv3x3(in_planes=num_channels[0], out_planes=num_channels[0]))
            layers.append(nn.BatchNorm2d(num_channels[0]))

            layers.append(conv1x1(in_planes=num_channels[0], out_planes=output_dim[i], stride=1))

            self.task.append(nn.Sequential(*layers))

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
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # Encoder stage 1
        identity = x
        x = self.relu(self.encoder_stage1_bn1(self.encoder_stage1_conv1(x)))
        x = self.relu(self.encoder_stage1_bn2(self.encoder_stage1_conv2(x)) + identity)
        identity = x
        x = self.relu(self.encoder_stage1_bn3(self.encoder_stage1_conv3(x)))
        x = self.relu(self.encoder_stage1_bn4(self.encoder_stage1_conv4(x)) + identity)

        # Encoder stage 2
        identity = self.relu(self.downsample_bn12(self.downsample_conv12(x)))
        x = self.relu(self.encoder_stage2_bn1(self.encoder_stage2_conv1(x)))
        x = self.relu(self.encoder_stage2_bn2(self.encoder_stage2_conv2(x)) + identity)
        identity = x
        x = self.relu(self.encoder_stage2_bn3(self.encoder_stage2_conv3(x)))
        x = self.relu(self.encoder_stage2_bn4(self.encoder_stage2_conv4(x)) + identity)

        # Encoder stage 3
        identity = self.relu(self.downsample_bn23(self.downsample_conv23(x)))
        x = self.relu(self.encoder_stage3_bn1(self.encoder_stage3_conv1(x)))
        x = self.relu(self.encoder_stage3_bn2(self.encoder_stage3_conv2(x)) + identity)
        identity = x
        x = self.relu(self.encoder_stage3_bn3(self.encoder_stage3_conv3(x)))
        x = self.relu(self.encoder_stage3_bn4(self.encoder_stage3_conv4(x)) + identity)

        # Encoder stage 4
        identity = self.relu(self.downsample_bn34(self.downsample_conv34(x)))
        x = self.relu(self.encoder_stage4_bn1(self.encoder_stage4_conv1(x)))
        x = self.relu(self.encoder_stage4_bn2(self.encoder_stage4_conv2(x)) + identity)
        identity = x
        x = self.relu(self.encoder_stage4_bn3(self.encoder_stage4_conv3(x)))
        x = self.relu(self.encoder_stage4_bn4(self.encoder_stage4_conv4(x)) + identity)

        # Decoder stage 4
        identity = x
        x = self.relu(self.decoder_stage4_bn1(self.decoder_stage4_conv1(x)))
        x = self.relu(self.decoder_stage4_bn2(self.decoder_stage4_conv2(x)) + identity)
        identity = self.relu(self.upsample_bn43(self.upsample_conv43(x)))
        x = self.relu(self.decoder_stage4_bn3(self.decoder_stage4_conv3(x)))
        x = self.relu(self.decoder_stage4_bn4(self.decoder_stage4_conv4(x)) + identity)

        # Decoder stage 3
        identity = x
        x = self.relu(self.decoder_stage3_bn1(self.decoder_stage3_conv1(x)))
        x = self.relu(self.decoder_stage3_bn2(self.decoder_stage3_conv2(x)) + identity)
        identity = self.relu(self.upsample_bn32(self.upsample_conv32(x)))
        x = self.relu(self.decoder_stage3_bn3(self.decoder_stage3_conv3(x)))
        x = self.relu(self.decoder_stage3_bn4(self.decoder_stage3_conv4(x)) + identity)


        y = [0] * self.num_task
        for i in range(self.num_task):
            y[i] = self.task[i](x)

        return [y[i] for i in range(self.num_task)], self.logsigma
