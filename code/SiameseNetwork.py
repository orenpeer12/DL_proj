import os

import torch
from torch import nn
from torchvision import models
import numpy as np
from pathlib import Path
# import torch.functional as F
from torch.nn.init import kaiming_normal_
from pre_trained_models.resnet50_ft_pytorch.resnet50_ft_dims_2048 import resnet50_ft
from pre_trained_models.senet50_256_pytorch.senet50_256 import senet50_256

class SiameseNetwork(nn.Module):
    def __init__(self, model_time):
        super(SiameseNetwork, self).__init__()

        self.name = str(model_time)

        # load pre-trained resnet model
        root_folder = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        resnet50_model = resnet50_ft(
            root_folder / 'pre_trained_models' / 'resnet50_ft_pytorch' / 'resnet50_ft_dims_2048.pth')
        senet50_256_model = senet50_256(
            root_folder / 'pre_trained_models' / 'senet50_256_pytorch' / 'senet50_256.pth')

        pretrained_model = senet50_256_model

        self.features = pretrained_model

        # Load trained model for transfer learning:
        # self.model = models.vgg16_bn(pretrained=True)
        num_features = self.features.feat_extract.out_channels     # VGG
        print("features space size: {}".format(num_features))  # VGG
        # common part ('siamese')
        # self.model.classifier = self.model.classifier[:-1]
        # Separate part - 2 featurs_vectors -> one long vector -> classify:
        self.classifier = nn.Sequential(nn.Linear(2 * num_features, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, 32),
                                            nn.ReLU(),
                                            nn.Linear(32, 1),
                                            nn.Sigmoid()
                                            )
        for param in self.features.parameters():
            param.requires_grad = False

        # for i, param in enumerate(self.model.classifier.parameters()):
        #     param.requires_grad = False

        self.initialize()

        # self.model.classifier.add_module('Our Classifier', nn.Linear(2 * num_features, 1))
        # self.model.classifier.add_module('Our Sigmoid', nn.Sigmoid())

    def forward(self, input1, input2):
        feat1 = self.features(input1)[0]
        feat1 = feat1.view(feat1.size()[0], -1)  # make it suitable for fc layer.
        feat2 = self.features(input2)[0]
        feat2 = feat2.view(feat2.size()[0], -1)  # make it suitable for fc layer.
        # feat = feat1 + feat2
        feat = torch.cat((feat1, feat2), dim=1)
        output = self.classifier(feat)
        return output

    # init weights of our classifier
    def initialize(self):
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.zero_()
        # self.model.classifier[-2].apply(init_weights)
        self.classifier.apply(init_weights)
        trainable_model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        non_trainable_model_parameters = filter(lambda p: not(p.requires_grad), self.parameters())
        trainable_params = sum([np.prod(p.size()) for p in trainable_model_parameters])
        non_trainable_params = sum([np.prod(p.size()) for p in non_trainable_model_parameters])
        print("Num. of trainable parameters: {:,}, num. of frozen parameters: {:,}, total: {:,}".format(
            trainable_params, non_trainable_params, trainable_params + non_trainable_params))

    def train(self):
        self.features.eval()
        self.classifier.train()

    def eval(self):
        self.features.eval()
        self.classifier.eval()


class ConvBlock(nn.Module):

    def __init__(self, input_dim=128, n_filters=256, kernel_size=3, padding=1, stride=1, shortcut=False,
                 downsampling=None):
        super(ConvBlock, self).__init__()

        self.downsampling = downsampling
        self.shortcut = shortcut
        self.conv1 = nn.Conv1d(input_dim, n_filters, kernel_size=kernel_size, padding=padding, stride=stride)
        self.batchnorm1 = nn.BatchNorm1d(n_filters)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=padding, stride=stride)
        self.batchnorm2 = nn.BatchNorm1d(n_filters)
        self.relu2 = nn.ReLU()

    def forward(self, input):

        residual = input
        output = self.conv1(input)
        output = self.batchnorm1(output)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.batchnorm2(output)

        if self.shortcut:
            if self.downsampling is not None:
                residual = self.downsampling(input)
            output += residual

        output = self.relu2(output)

        return output


class VDCNN(nn.Module):

    def __init__(self, model_time, n_classes=2, num_embedding=69, embedding_dim=16, depth=9, n_fc_neurons=2048, shortcut=False):
        super(VDCNN, self).__init__()
        self.name = str(model_time)
        layers = []
        fc_layers = []
        base_num_features = 64

        # self.embed = nn.Embedding(num_embedding, embedding_dim, padding_idx=0, max_norm=None,
        #                           norm_type=2, scale_grad_by_freq=False, sparse=False)

        layers.append(nn.Conv1d(embedding_dim, base_num_features, kernel_size=3, padding=1))

        if depth == 9:
            num_conv_block = [0, 0, 0, 0]
        elif depth == 17:
            num_conv_block = [1, 1, 1, 1]
        elif depth == 29:
            num_conv_block = [4, 4, 1, 1]
        elif depth == 49:
            num_conv_block = [7, 7, 4, 2]

        layers.append(ConvBlock(input_dim=base_num_features, n_filters=base_num_features, kernel_size=3, padding=1,
                                shortcut=shortcut))
        for _ in range(num_conv_block[0]):
            layers.append(ConvBlock(input_dim=base_num_features, n_filters=base_num_features, kernel_size=3, padding=1,
                                    shortcut=shortcut))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        ds = nn.Sequential(nn.Conv1d(base_num_features, 2 * base_num_features, kernel_size=1, stride=1, bias=False),
                           nn.BatchNorm1d(2 * base_num_features))
        layers.append(
            ConvBlock(input_dim=base_num_features, n_filters=2 * base_num_features, kernel_size=3, padding=1,
                      shortcut=shortcut, downsampling=ds))
        for _ in range(num_conv_block[1]):
            layers.append(
                ConvBlock(input_dim=2 * base_num_features, n_filters=2 * base_num_features, kernel_size=3, padding=1,
                          shortcut=shortcut))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        ds = nn.Sequential(nn.Conv1d(2 * base_num_features, 4 * base_num_features, kernel_size=1, stride=1, bias=False),
                           nn.BatchNorm1d(4 * base_num_features))
        layers.append(
            ConvBlock(input_dim=2 * base_num_features, n_filters=4 * base_num_features, kernel_size=3, padding=1,
                      shortcut=shortcut, downsampling=ds))
        for _ in range(num_conv_block[2]):
            layers.append(
                ConvBlock(input_dim=4 * base_num_features, n_filters=4 * base_num_features, kernel_size=3, padding=1,
                          shortcut=shortcut))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        ds = nn.Sequential(nn.Conv1d(4 * base_num_features, 8 * base_num_features, kernel_size=1, stride=1, bias=False),
                           nn.BatchNorm1d(8 * base_num_features))
        layers.append(
            ConvBlock(input_dim=4 * base_num_features, n_filters=8 * base_num_features, kernel_size=3, padding=1,
                      shortcut=shortcut, downsampling=ds))
        for _ in range(num_conv_block[3]):
            layers.append(
                ConvBlock(input_dim=8 * base_num_features, n_filters=8 * base_num_features, kernel_size=3, padding=1,
                          shortcut=shortcut))

        layers.append(nn.AdaptiveMaxPool1d(8))
        fc_layers.extend([nn.Linear(8 * 8 * base_num_features, n_fc_neurons), nn.ReLU()])
        fc_layers.extend([nn.Linear(n_fc_neurons, n_fc_neurons), nn.ReLU()])
        fc_layers.extend([nn.Linear(n_fc_neurons, n_classes)])

        self.layers = nn.Sequential(*layers)
        self.fc_layers = nn.Sequential(*fc_layers)
        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, input1, input2):
        output = torch.cat((input1, input2), dim=2)
        # output = output.transpose(1, 2)
        output = self.layers(output)
        output = output.view(output.size(0), -1)
        output = self.fc_layers(output)

        return output


class SiameseNetwork2(nn.Module):

    def __init__(self):
        super(SiameseNetwork2, self).__init__()
        self.meta = {'mean': [129.186279296875, 104.76238250732422, 93.59396362304688],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3]}
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.fc6 = nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout7 = nn.Dropout(p=0.5)
        self.fc8 = nn.Linear(in_features=4096, out_features=2622, bias=True)
        self.fc9 = nn.Sequential(
            nn.Linear(2622, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 2))

    def forward_once(self, x0):
        x1 = self.conv1_1(x0)
        x2 = self.relu1_1(x1)
        x3 = self.conv1_2(x2)
        x4 = self.relu1_2(x3)
        x5 = self.pool1(x4)
        x6 = self.conv2_1(x5)
        x7 = self.relu2_1(x6)
        x8 = self.conv2_2(x7)
        x9 = self.relu2_2(x8)
        x10 = self.pool2(x9)
        x11 = self.conv3_1(x10)
        x12 = self.relu3_1(x11)
        x13 = self.conv3_2(x12)
        x14 = self.relu3_2(x13)
        x15 = self.conv3_3(x14)
        x16 = self.relu3_3(x15)
        x17 = self.pool3(x16)
        x18 = self.conv4_1(x17)
        x19 = self.relu4_1(x18)
        x20 = self.conv4_2(x19)
        x21 = self.relu4_2(x20)
        x22 = self.conv4_3(x21)
        x23 = self.relu4_3(x22)
        x24 = self.pool4(x23)
        x25 = self.conv5_1(x24)
        x26 = self.relu5_1(x25)
        x27 = self.conv5_2(x26)
        x28 = self.relu5_2(x27)
        x29 = self.conv5_3(x28)
        x30 = self.relu5_3(x29)
        x31_preflatten = self.pool5(x30)
        x31 = x31_preflatten.view(x31_preflatten.size(0), -1)
        x32 = self.fc6(x31)
        x33 = self.relu6(x32)
        x34 = self.dropout6(x33)
        x35 = self.fc7(x34)
        x36 = self.relu7(x35)
        x37 = self.dropout7(x36)
        x38 = self.fc8(x37)
        return x38


    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        difference = output1 - output2
        output = self.fc9(difference)
        return output


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)
# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        # self.fc = nn.Linear(1344, num_classes)
        self.fc1 = nn.Linear(3136, 256)
        # self.fc1 = nn.Linear(6272, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 1)
        # Print model parameters
        trainable_model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        non_trainable_model_parameters = filter(lambda p: not (p.requires_grad), self.parameters())
        trainable_params = sum([np.prod(p.size()) for p in trainable_model_parameters])
        non_trainable_params = sum([np.prod(p.size()) for p in non_trainable_model_parameters])
        print("Num. of trainable parameters: {:,}, num. of frozen parameters: {:,}, total: {:,}".format(
            trainable_params, non_trainable_params, trainable_params + non_trainable_params))

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward_common(self, input):  # siamese NN
        # feed input1
        out = self.conv(input)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, input1, input2):  # siamese NN
        # feed input1
        out1 = self.forward_common(input1)
        # feed input2
        out2 = self.forward_common(input2)

        # out = torch.cat((out1, out2), dim=1)
        out = out1 + out2
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

    # def forward(self, input1, input2):  # 2 images concatenated, then, feed-forward into 1 model.
    #     x = torch.cat((input1, input2), dim=2)
    #     out = self.conv(x)
    #     out = self.bn(out)
    #     out = self.relu(out)
    #     out = self.layer1(out)
    #     out = self.layer2(out)
    #     out = self.layer3(out)
    #     out = self.avg_pool(out)
    #     out = out.view(out.size(0), -1)
    #     out = self.fc1(out)
    #     out = self.fc2(out)
    #     out = self.fc3(out)
    #     return out
