# An implementation of same architecture of SegNet (modules) but Deconv Layers
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

def deconv3x3(in_planes, out_planes, stride=2):
    """3x3 transpose convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, output_padding=(1,1), bias=False)


class ShareNet(nn.Module):
    def __init__(self, num_channels=[32,64,128,256,256]):
        super(ShareNet, self).__init__()
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

    def conv_layer_ff(self, x, weights, conv_index, stride, bn_mean, bn_var, bn_mean_index, bn_var_index, bn_weight_index, bn_bias_index):
        x = F.conv2d(input=x, weight=weights[conv_index], bias=None, stride=stride, padding=1)
        x = F.batch_norm(input=x, running_mean=bn_mean[bn_mean_index],
                                  running_var=bn_var[bn_var_index],
                                  weight=weights[bn_weight_index], bias=weights[bn_bias_index], training=True)
        x = F.relu(x, inplace=True)

        return x

    def deconv_layer_ff(self, x, weights, deconv_index, stride, bn_mean, bn_var, bn_mean_index, bn_var_index, bn_weight_index, bn_bias_index):
        x = F.conv_transpose2d(input=x, weight=weights[deconv_index], bias=None, stride=stride, padding=1, output_padding=(1,1))
        x = F.batch_norm(input=x, running_mean=bn_mean[bn_mean_index],
                                  running_var=bn_var[bn_var_index],
                                  weight=weights[bn_weight_index], bias=weights[bn_bias_index], training=True)
        x = F.relu(x, inplace=True)

        return x

    def forward(self, x, weights=None, bn_mean=None, bn_var=None):
        if weights is None:
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

        else:
            x = self.conv_layer_ff(x, weights, 'encoder_conv11.weight', 1, bn_mean, bn_var, 'encoder_bn11.running_mean', 'encoder_bn11.running_var', 'encoder_bn11.weight', 'encoder_bn11.bias')
            x = self.conv_layer_ff(x, weights, 'encoder_conv12.weight', 1, bn_mean, bn_var, 'encoder_bn12.running_mean', 'encoder_bn12.running_var', 'encoder_bn12.weight', 'encoder_bn12.bias')
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            x = self.conv_layer_ff(x, weights, 'encoder_conv21.weight', 1, bn_mean, bn_var, 'encoder_bn21.running_mean', 'encoder_bn21.running_var', 'encoder_bn21.weight', 'encoder_bn21.bias')
            x = self.conv_layer_ff(x, weights, 'encoder_conv22.weight', 1, bn_mean, bn_var, 'encoder_bn22.running_mean', 'encoder_bn22.running_var', 'encoder_bn22.weight', 'encoder_bn22.bias')
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            x = self.conv_layer_ff(x, weights, 'encoder_conv31.weight', 1, bn_mean, bn_var, 'encoder_bn31.running_mean', 'encoder_bn31.running_var', 'encoder_bn31.weight', 'encoder_bn31.bias')
            x = self.conv_layer_ff(x, weights, 'encoder_conv32.weight', 1, bn_mean, bn_var, 'encoder_bn32.running_mean', 'encoder_bn32.running_var', 'encoder_bn32.weight', 'encoder_bn32.bias')
            x = self.conv_layer_ff(x, weights, 'encoder_conv33.weight', 1, bn_mean, bn_var, 'encoder_bn33.running_mean', 'encoder_bn33.running_var', 'encoder_bn33.weight', 'encoder_bn33.bias')
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            x = self.conv_layer_ff(x, weights, 'encoder_conv41.weight', 1, bn_mean, bn_var, 'encoder_bn41.running_mean', 'encoder_bn41.running_var', 'encoder_bn41.weight', 'encoder_bn41.bias')
            x = self.conv_layer_ff(x, weights, 'encoder_conv42.weight', 1, bn_mean, bn_var, 'encoder_bn42.running_mean', 'encoder_bn42.running_var', 'encoder_bn42.weight', 'encoder_bn42.bias')
            x = self.conv_layer_ff(x, weights, 'encoder_conv43.weight', 1, bn_mean, bn_var, 'encoder_bn43.running_mean', 'encoder_bn43.running_var', 'encoder_bn43.weight', 'encoder_bn43.bias')
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            x = self.conv_layer_ff(x, weights, 'encoder_conv51.weight', 1, bn_mean, bn_var, 'encoder_bn51.running_mean', 'encoder_bn51.running_var', 'encoder_bn51.weight', 'encoder_bn51.bias')
            x = self.conv_layer_ff(x, weights, 'encoder_conv52.weight', 1, bn_mean, bn_var, 'encoder_bn52.running_mean', 'encoder_bn52.running_var', 'encoder_bn52.weight', 'encoder_bn52.bias')
            x = self.conv_layer_ff(x, weights, 'encoder_conv53.weight', 1, bn_mean, bn_var, 'encoder_bn53.running_mean', 'encoder_bn53.running_var', 'encoder_bn53.weight', 'encoder_bn53.bias')
            x = F.max_pool2d(x, kernel_size=2, stride=2)

            x = self.deconv_layer_ff(x, weights, 'decoder_deconv1.weight', 2, bn_mean, bn_var, 'decoder_bn1.running_mean', 'decoder_bn1.running_var', 'decoder_bn1.weight', 'decoder_bn1.bias')
            x = self.conv_layer_ff(x, weights, 'decoder_conv11.weight', 1, bn_mean, bn_var, 'decoder_bn11.running_mean', 'decoder_bn11.running_var', 'decoder_bn11.weight', 'decoder_bn11.bias')
            x = self.conv_layer_ff(x, weights, 'decoder_conv12.weight', 1, bn_mean, bn_var, 'decoder_bn12.running_mean', 'decoder_bn12.running_var', 'decoder_bn12.weight', 'decoder_bn12.bias')

            x = self.deconv_layer_ff(x, weights, 'decoder_deconv2.weight', 2, bn_mean, bn_var, 'decoder_bn2.running_mean', 'decoder_bn2.running_var', 'decoder_bn2.weight', 'decoder_bn2.bias')
            x = self.conv_layer_ff(x, weights, 'decoder_conv21.weight', 1, bn_mean, bn_var, 'decoder_bn21.running_mean', 'decoder_bn21.running_var', 'decoder_bn21.weight', 'decoder_bn21.bias')
            x = self.conv_layer_ff(x, weights, 'decoder_conv22.weight', 1, bn_mean, bn_var, 'decoder_bn22.running_mean', 'decoder_bn22.running_var', 'decoder_bn22.weight', 'decoder_bn22.bias')

            x = self.deconv_layer_ff(x, weights, 'decoder_deconv3.weight', 2, bn_mean, bn_var, 'decoder_bn3.running_mean', 'decoder_bn3.running_var', 'decoder_bn3.weight', 'decoder_bn3.bias')
            x = self.conv_layer_ff(x, weights, 'decoder_conv31.weight', 1, bn_mean, bn_var, 'decoder_bn31.running_mean', 'decoder_bn31.running_var', 'decoder_bn31.weight', 'decoder_bn31.bias')
            x = self.conv_layer_ff(x, weights, 'decoder_conv32.weight', 1, bn_mean, bn_var, 'decoder_bn32.running_mean', 'decoder_bn32.running_var', 'decoder_bn32.weight', 'decoder_bn32.bias')

            x = self.deconv_layer_ff(x, weights, 'decoder_deconv4.weight', 2, bn_mean, bn_var, 'decoder_bn4.running_mean', 'decoder_bn4.running_var', 'decoder_bn4.weight', 'decoder_bn4.bias')
            x = self.conv_layer_ff(x, weights, 'decoder_conv41.weight', 1, bn_mean, bn_var, 'decoder_bn41.running_mean', 'decoder_bn41.running_var', 'decoder_bn41.weight', 'decoder_bn41.bias')

        return x


class BranchNet(nn.Module):
    def __init__(self, output_dim=13, num_channels=[32,64,128,256,256]):
        super(BranchNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.output_dim = output_dim

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
        x = self.relu(self.bn5(self.deconv5(x)))
        x = self.relu(self.bn51(self.conv51(x)))
        x = self.pred(x)

        return x
