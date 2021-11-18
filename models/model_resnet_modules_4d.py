import torch.nn.functional as F
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


class ShareNet(nn.Module):
    def __init__(self, num_channels=[32,64,128,256]):
        super(ShareNet, self).__init__()
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

    def conv_layer_ff(self, x, weights, conv_index, stride, padding, bn_mean, bn_var, bn_mean_index, bn_var_index, bn_weight_index, bn_bias_index, identity=None):
        x = F.conv2d(input=x, weight=weights[conv_index], bias=None, stride=stride, padding=padding)
        x = F.batch_norm(input=x, running_mean=bn_mean[bn_mean_index],
                                  running_var=bn_var[bn_var_index],
                                  weight=weights[bn_weight_index], bias=weights[bn_bias_index], training=True)
        if identity is not None:
            x = F.relu(x+identity, inplace=True)
        else:
            x = F.relu(x, inplace=True)

        return x

    def deconv_layer_ff(self, x, weights, deconv_index, stride, padding, bn_mean, bn_var, bn_mean_index, bn_var_index, bn_weight_index, bn_bias_index, identity=None):
        x = F.conv_transpose2d(input=x, weight=weights[deconv_index], bias=None, stride=stride, padding=padding, output_padding=(1,1))
        x = F.batch_norm(input=x, running_mean=bn_mean[bn_mean_index],
                                  running_var=bn_var[bn_var_index],
                                  weight=weights[bn_weight_index], bias=weights[bn_bias_index], training=True)
        if identity is not None:
            x = F.relu(x+identity, inplace=True)
        else:
            x = F.relu(x, inplace=True)

        return x

    def forward(self, x, weights=None, bn_mean=None, bn_var=None):
        if weights is None:
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

        else:
            x = self.conv_layer_ff(x, weights, 'conv1.weight', 2, 3, bn_mean, bn_var, 'bn1.running_mean', 'bn1.running_var', 'bn1.weight', 'bn1.bias')
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

            # Encoder stage 1
            identity = x
            x = self.conv_layer_ff(x, weights, 'encoder_stage1_conv1.weight', 1, 1, bn_mean, bn_var, 'encoder_stage1_bn1.running_mean', 'encoder_stage1_bn1.running_var', 'encoder_stage1_bn1.weight', 'encoder_stage1_bn1.bias', None)
            x = self.conv_layer_ff(x, weights, 'encoder_stage1_conv2.weight', 1, 1, bn_mean, bn_var, 'encoder_stage1_bn2.running_mean', 'encoder_stage1_bn2.running_var', 'encoder_stage1_bn2.weight', 'encoder_stage1_bn2.bias', identity)
            identity = x
            x = self.conv_layer_ff(x, weights, 'encoder_stage1_conv3.weight', 1, 1, bn_mean, bn_var, 'encoder_stage1_bn3.running_mean', 'encoder_stage1_bn3.running_var', 'encoder_stage1_bn3.weight', 'encoder_stage1_bn3.bias', None)
            x = self.conv_layer_ff(x, weights, 'encoder_stage1_conv4.weight', 1, 1, bn_mean, bn_var, 'encoder_stage1_bn4.running_mean', 'encoder_stage1_bn4.running_var', 'encoder_stage1_bn4.weight', 'encoder_stage1_bn4.bias', identity)

            # Encoder stage 2
            identity = self.conv_layer_ff(x, weights, 'downsample_conv12.weight', 2, 0, bn_mean, bn_var, 'downsample_bn12.running_mean', 'downsample_bn12.running_var', 'downsample_bn12.weight', 'downsample_bn12.bias', None)
            x = self.conv_layer_ff(x, weights, 'encoder_stage2_conv1.weight', 2, 1, bn_mean, bn_var, 'encoder_stage2_bn1.running_mean', 'encoder_stage2_bn1.running_var', 'encoder_stage2_bn1.weight', 'encoder_stage2_bn1.bias', None)
            x = self.conv_layer_ff(x, weights, 'encoder_stage2_conv2.weight', 1, 1, bn_mean, bn_var, 'encoder_stage2_bn2.running_mean', 'encoder_stage2_bn2.running_var', 'encoder_stage2_bn2.weight', 'encoder_stage2_bn2.bias', identity)
            identity = x
            x = self.conv_layer_ff(x, weights, 'encoder_stage2_conv3.weight', 1, 1, bn_mean, bn_var, 'encoder_stage2_bn3.running_mean', 'encoder_stage2_bn3.running_var', 'encoder_stage2_bn3.weight', 'encoder_stage2_bn3.bias', None)
            x = self.conv_layer_ff(x, weights, 'encoder_stage2_conv4.weight', 1, 1, bn_mean, bn_var, 'encoder_stage2_bn4.running_mean', 'encoder_stage2_bn4.running_var', 'encoder_stage2_bn4.weight', 'encoder_stage2_bn4.bias', identity)

            # Encoder stage 3
            identity = self.conv_layer_ff(x, weights, 'downsample_conv23.weight', 2, 0, bn_mean, bn_var, 'downsample_bn23.running_mean', 'downsample_bn23.running_var', 'downsample_bn23.weight', 'downsample_bn23.bias', None)
            x = self.conv_layer_ff(x, weights, 'encoder_stage3_conv1.weight', 2, 1, bn_mean, bn_var, 'encoder_stage3_bn1.running_mean', 'encoder_stage3_bn1.running_var', 'encoder_stage3_bn1.weight', 'encoder_stage3_bn1.bias', None)
            x = self.conv_layer_ff(x, weights, 'encoder_stage3_conv2.weight', 1, 1, bn_mean, bn_var, 'encoder_stage3_bn2.running_mean', 'encoder_stage3_bn2.running_var', 'encoder_stage3_bn2.weight', 'encoder_stage3_bn2.bias', identity)
            identity = x
            x = self.conv_layer_ff(x, weights, 'encoder_stage3_conv3.weight', 1, 1, bn_mean, bn_var, 'encoder_stage3_bn3.running_mean', 'encoder_stage3_bn3.running_var', 'encoder_stage3_bn3.weight', 'encoder_stage3_bn3.bias', None)
            x = self.conv_layer_ff(x, weights, 'encoder_stage3_conv4.weight', 1, 1, bn_mean, bn_var, 'encoder_stage3_bn4.running_mean', 'encoder_stage3_bn4.running_var', 'encoder_stage3_bn4.weight', 'encoder_stage3_bn4.bias', identity)

            # Encoder stage 4
            identity = self.conv_layer_ff(x, weights, 'downsample_conv34.weight', 2, 0, bn_mean, bn_var, 'downsample_bn34.running_mean', 'downsample_bn34.running_var', 'downsample_bn34.weight', 'downsample_bn34.bias', None)
            x = self.conv_layer_ff(x, weights, 'encoder_stage4_conv1.weight', 2, 1, bn_mean, bn_var, 'encoder_stage4_bn1.running_mean', 'encoder_stage4_bn1.running_var', 'encoder_stage4_bn1.weight', 'encoder_stage4_bn1.bias', None)
            x = self.conv_layer_ff(x, weights, 'encoder_stage4_conv2.weight', 1, 1, bn_mean, bn_var, 'encoder_stage4_bn2.running_mean', 'encoder_stage4_bn2.running_var', 'encoder_stage4_bn2.weight', 'encoder_stage4_bn2.bias', identity)
            identity = x
            x = self.conv_layer_ff(x, weights, 'encoder_stage4_conv3.weight', 1, 1, bn_mean, bn_var, 'encoder_stage4_bn3.running_mean', 'encoder_stage4_bn3.running_var', 'encoder_stage4_bn3.weight', 'encoder_stage4_bn3.bias', None)
            x = self.conv_layer_ff(x, weights, 'encoder_stage4_conv4.weight', 1, 1, bn_mean, bn_var, 'encoder_stage4_bn4.running_mean', 'encoder_stage4_bn4.running_var', 'encoder_stage4_bn4.weight', 'encoder_stage4_bn4.bias', identity)

            # decoder stage 4
            identity = x
            x = self.conv_layer_ff(x, weights, 'decoder_stage4_conv1.weight', 1, 1, bn_mean, bn_var, 'decoder_stage4_bn1.running_mean', 'decoder_stage4_bn1.running_var', 'decoder_stage4_bn1.weight', 'decoder_stage4_bn1.bias', None)
            x = self.conv_layer_ff(x, weights, 'decoder_stage4_conv2.weight', 1, 1, bn_mean, bn_var, 'decoder_stage4_bn2.running_mean', 'decoder_stage4_bn2.running_var', 'decoder_stage4_bn2.weight', 'decoder_stage4_bn2.bias', identity)
            identity = self.deconv_layer_ff(x, weights, 'upsample_conv43.weight', 2, 0, bn_mean, bn_var, 'upsample_bn43.running_mean', 'upsample_bn43.running_var', 'upsample_bn43.weight', 'upsample_bn43.bias', None)
            x = self.conv_layer_ff(x, weights, 'decoder_stage4_conv3.weight', 1, 1, bn_mean, bn_var, 'decoder_stage4_bn3.running_mean', 'decoder_stage4_bn3.running_var', 'decoder_stage4_bn3.weight', 'decoder_stage4_bn3.bias', None)
            x = self.deconv_layer_ff(x, weights, 'decoder_stage4_conv4.weight', 2, 1, bn_mean, bn_var, 'decoder_stage4_bn4.running_mean', 'decoder_stage4_bn4.running_var', 'decoder_stage4_bn4.weight', 'decoder_stage4_bn4.bias', identity)

        return x


class BranchNet(nn.Module):
    def __init__(self, output_dim=13, num_channels=[32,64,128,256]):
        super(BranchNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.output_dim = output_dim

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

        # Decoder stage 2
        self.decoder_stage2_conv1 = conv3x3(in_planes=num_channels[1], out_planes=num_channels[1])
        self.decoder_stage2_bn1 = nn.BatchNorm2d(num_channels[1])
        self.decoder_stage2_conv2 = conv3x3(in_planes=num_channels[1], out_planes=num_channels[1])
        self.decoder_stage2_bn2 = nn.BatchNorm2d(num_channels[1])

        self.decoder_stage2_conv3 = conv3x3(in_planes=num_channels[1], out_planes=num_channels[1])
        self.decoder_stage2_bn3 = nn.BatchNorm2d(num_channels[1])
        self.decoder_stage2_conv4 = deconv3x3(in_planes=num_channels[1], out_planes=num_channels[0], stride=2)
        self.decoder_stage2_bn4 = nn.BatchNorm2d(num_channels[0])

        # Decoder stage 1
        self.decoder_stage1_conv1 = conv3x3(in_planes=num_channels[0], out_planes=num_channels[0])
        self.decoder_stage1_bn1 = nn.BatchNorm2d(num_channels[0])
        self.decoder_stage1_conv2 = conv3x3(in_planes=num_channels[0], out_planes=num_channels[0])
        self.decoder_stage1_bn2 = nn.BatchNorm2d(num_channels[0])
        self.decoder_stage1_conv3 = conv3x3(in_planes=num_channels[0], out_planes=num_channels[0])
        self.decoder_stage1_bn3 = nn.BatchNorm2d(num_channels[0])
        self.decoder_stage1_conv4 = conv3x3(in_planes=num_channels[0], out_planes=num_channels[0])
        self.decoder_stage1_bn4 = nn.BatchNorm2d(num_channels[0])

        self.deconv4 = deconv3x3(in_planes=num_channels[0], out_planes=num_channels[0], stride=2)
        self.bn4 = nn.BatchNorm2d(num_channels[0])
        self.conv41 = conv3x3(in_planes=num_channels[0], out_planes=num_channels[0])
        self.bn41 = nn.BatchNorm2d(num_channels[0])

        self.deconv5 = deconv3x3(in_planes=num_channels[0], out_planes=num_channels[0], stride=2)
        self.bn5 = nn.BatchNorm2d(num_channels[0])
        self.conv51 = conv3x3(in_planes=num_channels[0], out_planes=num_channels[0])
        self.bn51 = nn.BatchNorm2d(num_channels[0])

        self.pred = conv1x1(in_planes=num_channels[0], out_planes=self.output_dim, stride=1)

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
        # Decoder stage 3
        x = self.relu(self.decoder_stage3_bn1(self.decoder_stage3_conv1(x)))
        x = self.relu(self.decoder_stage3_bn2(self.decoder_stage3_conv2(x)))
        x = self.relu(self.decoder_stage3_bn3(self.decoder_stage3_conv3(x)))
        x = self.relu(self.decoder_stage3_bn4(self.decoder_stage3_conv4(x)))

        # Decoder stage 2
        x = self.relu(self.decoder_stage2_bn1(self.decoder_stage2_conv1(x)))
        x = self.relu(self.decoder_stage2_bn2(self.decoder_stage2_conv2(x)))
        x = self.relu(self.decoder_stage2_bn3(self.decoder_stage2_conv3(x)))
        x = self.relu(self.decoder_stage2_bn4(self.decoder_stage2_conv4(x)))

        # Decoder stage 1
        x = self.relu(self.decoder_stage1_bn1(self.decoder_stage1_conv1(x)))
        x = self.relu(self.decoder_stage1_bn2(self.decoder_stage1_conv2(x)))
        x = self.relu(self.decoder_stage1_bn3(self.decoder_stage1_conv3(x)))
        x = self.relu(self.decoder_stage1_bn4(self.decoder_stage1_conv4(x)))

        x = self.relu(self.bn4(self.deconv4(x)))
        x = self.relu(self.bn41(self.conv41(x)))
        x = self.relu(self.bn5(self.deconv5(x)))
        x = self.relu(self.bn51(self.conv51(x)))
        x = self.pred(x)

        return x
