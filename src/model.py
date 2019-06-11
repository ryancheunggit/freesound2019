"""Model zoo."""
# BNVgg8 adopted from public kernel https://www.kaggle.com/mhiro2/simple-2d-cnn-classifier-with-pytorch
# Dual Path Networs, adopted from https://github.com/oyam/pytorch-DPNs/blob/master/dpn.py.

import torch
from collections import OrderedDict
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torchvision import models


class MeanMaxPool(nn.Module):
    def __init__(self):
        super(MeanMaxPool, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(None, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d(output_size=1)

    def forward(self, x):
        x = self.global_max_pool(self.global_avg_pool(x))
        x = x.view(x.shape[0], -1)
        return x


class MaxAvgPool(nn.Module):
    def __init__(self):
        super(MaxAvgPool, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(output_size=1)

    def forward(self, x):
        x = 0.5 * (self.global_avg_pool(x) + self.global_max_pool(x))
        x = x.view(x.shape[0], -1)
        return x


class ConcatPool(nn.Module):
    def __init__(self):
        super(ConcatPool, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(output_size=1)

    def forward(self, x):
        x = torch.cat([self.global_avg_pool(x), self.global_max_pool(x)], 1)
        x = x.view(x.shape[0], -1)
        return x


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class LastLinear(nn.Module):
    def __init__(self, in_features, hidden_size=256, dropout_rate=0.2):
        super(LastLinear, self).__init__()
        self.lastliner = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_size, bias=True),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=80)
        )

    def forward(self, x):
        return self.lastliner(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool_type='max', use_bn=True):
        super(ConvBlock, self).__init__()
        assert pool_type in ['avg', 'max'], Exception('Unaccepted pool_type, must be avg or max')
        self.pool_type = pool_type
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=(kernel_size, kernel_size), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=(kernel_size, kernel_size), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.init_weights()

    def _init_layer(self, layer, nonlinearity='leaky_relu'):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)

    def _init_bn(self, bn):
        bn.bias.data.fill_(0.)
        bn.running_mean.data.fill_(0.)
        bn.weight.data.fill_(1.)
        bn.running_var.data.fill_(1.)

    def init_weights(self):
        self._init_layer(self.conv1)
        self._init_layer(self.conv2)
        self._init_bn(self.bn1)
        self._init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2)):
        if self.use_bn:
            x = F.relu(self.bn1(self.conv1(input)))
            x = F.relu(self.bn2(self.conv2(x)))
        else:
            x = F.relu(self.conv1(input))
            x = F.relu(self.conv2(x))
        if self.pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif self.pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        return x


class DPConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DPConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3),
                               stride=(stride, stride), padding=(1, 1), groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                               stride=(1, 1), padding=(0, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class DualPathBlock(nn.Module):
    def __init__(self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, groups, block_type='normal'):
        super(DualPathBlock, self).__init__()
        self.num_1x1_c = num_1x1_c

        if block_type == 'proj':
            key_stride = 1
            self.has_proj = True

        if block_type == 'down':
            key_stride = 2
            self.has_proj = True

        if block_type == 'normal':
            key_stride = 1
            self.has_proj = False

        if self.has_proj:
            self.c1x1_w = self.BN_ReLU_Conv(in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1,
                                            stride=key_stride)
        self.layers = nn.Sequential(OrderedDict([
            ('c1x1_a', self.BN_ReLU_Conv(in_chs=in_chs, out_chs=num_1x1_a, kernel_size=1, stride=1)),
            ('c3x3_b', self.BN_ReLU_Conv(in_chs=num_1x1_a, out_chs=num_3x3_b, kernel_size=3, stride=key_stride,
                                         padding=1, groups=groups)),
            ('c1x1_c', self.BN_ReLU_Conv(in_chs=num_3x3_b, out_chs=num_1x1_c + inc, kernel_size=1, stride=1))
        ]))

    def BN_ReLU_Conv(self, in_chs, out_chs, kernel_size, stride, padding=0, groups=1):
        return nn.Sequential(OrderedDict([
            ('norm', nn.BatchNorm2d(in_chs)),
            ('relu', nn.ReLU(inplace=True)),
            ('conv', nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, groups=groups, bias=False))
        ]))

    def forward(self, x):
        data_in = torch.cat(x, dim=1) if isinstance(x, list) else x
        if self.has_proj:
            data_o = self.c1x1_w(data_in)
            data_o1 = data_o[:, :self.num_1x1_c, :, :]
            data_o2 = data_o[:, self.num_1x1_c:, :, :]
        else:
            data_o1 = x[0]
            data_o2 = x[1]

        out = self.layers(data_in)
        summ = data_o1 + out[:, :self.num_1x1_c, :, :]
        dense = torch.cat([data_o2, out[:, self.num_1x1_c:, :, :]], dim=1)
        return [summ, dense]


class DPN(nn.Module):
    def __init__(self, input_channels=3, num_init_features=64, k_r=96, groups=32, k_sec=(3, 4, 20, 3),
                 inc_sec=(16, 32, 24, 128), num_classes=80):
        super(DPN, self).__init__()
        blocks = OrderedDict()

        blocks['conv1'] = nn.Sequential(
            nn.Conv2d(input_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        bw = 256
        inc = inc_sec[0]
        r = int((k_r * bw) / 256)
        blocks['conv2_1'] = DualPathBlock(num_init_features, r, r, bw, inc, groups, 'proj')
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[0] + 1):
            blocks['conv2_{}'.format(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal')
            in_chs += inc

        bw = 512
        inc = inc_sec[1]
        r = int((k_r * bw) / 256)
        blocks['conv3_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down')
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[1] + 1):
            blocks['conv3_{}'.format(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal')
            in_chs += inc

        bw = 1024
        inc = inc_sec[2]
        r = int((k_r * bw) / 256)
        blocks['conv4_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down')
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[2] + 1):
            blocks['conv4_{}'.format(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal')
            in_chs += inc

        bw = 2048
        inc = inc_sec[3]
        r = int((k_r * bw) / 256)
        blocks['conv5_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down')
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[3] + 1):
            blocks['conv5_{}'.format(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal')
            in_chs += inc

        self.features = nn.Sequential(blocks)
        self.pool = MeanMaxPool()
        self.classifier = LastLinear(in_features=2688)

    def forward(self, x):
        x = torch.cat(self.features(x), dim=1)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def resnet34(input_shape=(1, 128, 128)):
    model = models.resnet34(pretrained=False)
    model.conv1 = nn.Conv2d(
        in_channels=input_shape[0],
        out_channels=64,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        bias=False
    )
    model.avgpool = MeanMaxPool()
    model.fc = nn.Linear(in_features=512, out_features=80)
    return model


def resnet50(input_shape=(1, 128, 128)):
    model = models.resnet50(pretrained=False)
    model.conv1 = nn.Conv2d(
        in_channels=input_shape[0],
        out_channels=64,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        bias=False
    )
    model.avgpool = MeanMaxPool()
    model.fc = nn.Linear(in_features=2048, out_features=80)
    return model


def resnet101(input_shape=(1, 128, 128)):
    model = models.resnet101(pretrained=False)
    model.conv1 = nn.Conv2d(
        in_channels=input_shape[0],
        out_channels=64,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        bias=False
    )
    model.avgpool = MeanMaxPool()
    model.fc = nn.Linear(in_features=2048, out_features=80)
    return model


def densenet121(input_shape=(1, 128, 128)):
    model = models.densenet121(pretrained=False)
    model.features.conv0 = nn.Conv2d(
        in_channels=input_shape[0],
        out_channels=64,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        bias=False
    )

    model.classifier = nn.Linear(in_features=1024, out_features=80)
    return model


def densenet161(input_shape=(1, 128, 128)):
    model = models.densenet161(pretrained=False)
    model.features.conv0 = nn.Conv2d(
        in_channels=input_shape[0],
        out_channels=96,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        bias=False
    )
    model.classifier = nn.Linear(in_features=2208, out_features=80)
    return model


def densenet169(input_shape=(1, 128, 128)):
    model = models.densenet169(pretrained=False)
    model.features.conv0 = nn.Conv2d(
        in_channels=input_shape[0],
        out_channels=64,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        bias=False
    )
    model.classifier = nn.Linear(in_features=1664, out_features=80)
    return model


class BNVgg8(nn.Module):
    def __init__(self, input_shape=(1, 128, 128), pool_type='max'):
        super(BNVgg8, self).__init__()
        self.conv_block1 = ConvBlock(in_channels=input_shape[0], out_channels=64, pool_type=pool_type)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128, pool_type=pool_type)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256, pool_type=pool_type)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512, pool_type=pool_type)
        self.feature = MeanMaxPool()
        self.classifier = nn.Linear(in_features=512, out_features=80)

    def forward(self, x):
        x = self.conv_block1(x, pool_size=(2, 2))
        x = self.conv_block2(x, pool_size=(2, 2))
        x = self.conv_block3(x, pool_size=(2, 2))
        x = self.conv_block4(x, pool_size=(1, 1))
        x = self.feature(x)
        output = self.classifier(x)
        return output


def bnvgg13(input_shape=(1, 128, 128)):
    model = models.vgg13_bn(pretrained=False)
    model.features[0] = nn.Conv2d(
        in_channels=input_shape[0],
        out_channels=64,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        bias=False
    )
    model.avgpool = MeanMaxPool()
    model.classifier = nn.Linear(in_features=512, out_features=80)
    return model


class MobileNet(nn.Module):
    cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]

    def __init__(self, input_shape=(1, 128, 128)):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layers = self._make_layers(in_channels=16)
        self.pool = MeanMaxPool()
        self.linear = LastLinear(in_features=1024)

    def _make_layers(self, in_channels):
        layers = []
        for x in self.cfg:
            out_channels = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(DPConvBlock(in_channels, out_channels, stride))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layers(x)
        x = self.pool(x)
        x = self.linear(x)
        return x


def dpn92(input_shape=(1, 128, 128)):
    return DPN(input_channels=input_shape[0], num_init_features=64, k_r=96, groups=32, k_sec=(3, 4, 20, 3),
               inc_sec=(16, 32, 24, 128), num_classes=80)


def dpn98(input_shape):
    return DPN(input_channels=input_shape[0], num_init_features=96, k_r=160, groups=40, k_sec=(3, 6, 20, 3),
               inc_sec=(16, 32, 32, 128), num_classes=80)


def dpn107(input_shape):
    return DPN(input_channels=input_shape[0], num_init_features=128, k_r=200, groups=50, k_sec=(4, 8, 20, 3),
               inc_sec=(20, 64, 64, 128), num_classes=80)


def dpn131(input_shape):
    return DPN(input_channels=input_shape[0], num_init_features=128, k_r=160, groups=40, k_sec=(4, 8, 28, 3),
               inc_sec=(16, 32, 32, 128), num_classes=80)


class ZOO(object):
    menu = {
        'mobilenet': MobileNet,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'densenet121': densenet121,
        'densenet161': densenet161,
        'densenet169': densenet169,
        'bnvgg8': BNVgg8,
        'bnvgg13': bnvgg13,
        'dpn92': dpn92,
        'dpn98': dpn98,
        'dpn107': dpn107,
        'dpn131': dpn131
    }

    def __init__(self, model_name=None):
        self.model_constructor = self.menu[model_name]

    def __call__(self, input_shape):
        return self.model_constructor(input_shape)


if __name__ == '__main__':
    def test_model_constructors():
        input_shape = (3, 128, 128)
        for input_shape in [(1, 128, 128), (2, 128, 128), (3, 128, 128)]:
            n_channel, n_time_step, n_feature = input_shape
            x = Variable(torch.randn(32, n_channel, n_time_step, n_feature))
            for model_name in ZOO.menu.keys():
                model = ZOO(model_name)(input_shape)
                out = model(x)
                # print(input_shape, model_name, out.shape)
                assert out.shape == (32, 80), "model {} input shape {}".format(model_name, input_shape)

    test_model_constructors()
