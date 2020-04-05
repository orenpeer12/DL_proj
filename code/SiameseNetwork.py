import os
import torch
from torch import nn
from torchvision import models
import numpy as np
from pathlib import Path
# import torch.functional as F
from torch.nn.init import kaiming_normal_
from pre_trained_models.resnet50_ft_pytorch.resnet50_ft_dims_2048 import resnet50_ft, Resnet50_ft
from pre_trained_models.resnet50_128_pytorch.resnet50_128 import resnet50_128, Resnet50_128
from pre_trained_models.senet50_256_pytorch.senet50_256 import senet50_256, Senet50_256
from utils import *


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class SiameseNetwork(nn.Module):
    def __init__(self, model_name, hyper_params=None):
        super(SiameseNetwork, self).__init__()

        self.name = model_name

        if hyper_params is None:
            self.dropout_rate = 0.2
        else:
            self.dropout_rate = hyper_params["dropout_rate"]

        # load pre-trained resnet model
        root_folder = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        resnet50_model = resnet50_ft(
            root_folder / 'pre_trained_models_weights' / 'resnet50_ft_pytorch' / 'resnet50_ft_dims_2048.pth')
        resnet50_128_model = resnet50_128(
            root_folder / 'pre_trained_models_weights' / 'resnet50_128_pytorch' / 'resnet50_128.pth')
        senet50_256_model = senet50_256(
            root_folder / 'pre_trained_models_weights' / 'senet50_256_pytorch' / 'senet50_256.pth')

        pretrained_model = resnet50_model
        # pretrained_model = resnet50_128_model
        # pretrained_model = senet50_256_model

        self.features = pretrained_model
        # throw away last layer ("classifier") in resnet50:
        if isinstance(self.features, Resnet50_ft):
            num_features = self.features.classifier.in_channels
            self.features._modules.popitem()

        elif isinstance(self.features, Resnet50_128):
            num_features = 128
        elif isinstance(self.features, Senet50_256):
            num_features = 256
        print("features space size: {}".format(num_features))
        # common part ('siamese')
        # self.model.classifier = self.model.classifier[:-1]
        # Separate part - 2 featurs_vectors -> one long vector -> classify:

        classfier_input_size = 2 * 3 * num_features

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features=classfier_input_size),
            nn.Linear(classfier_input_size, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.BatchNorm1d(num_features=64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.BatchNorm1d(num_features=32),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.fla = Flatten()
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.mp = nn.AdaptiveMaxPool2d((1, 1))

        # featu res_layers = self.features.modules()
        # num_layers = len(features_layers)
        # num_trainable_layers = 3

        # for x in features_layers[:-num_trainable_layers]:
        #     x.requires_grad = False
        # for x in features_layers[-num_trainable_layers:]:
        #     x.requires_grad = True
        #     print(x, ": Not frozen!")

        # for i, (name, param) in enumerate(features_layers):
        #     print(name)
        #     if i < num_layers - num_trainable_layers:
        #         param.requires_grad = False
        #     else:
        #         param.requires_grad = True
        #         print(i, name, ": Not frozen!")

        for param in self.features.parameters():
            param.requires_grad = True
        # for param in self.features.classifier.parameters():
        #     param.requires_grad = False

        # for i, param in enumerate(self.model.classifier.parameters()):
        #     param.requires_grad = False

        self.initialize()

        # self.model.classifier.add_module('Our Classifier', nn.Linear(2 * num_features, 1))
        # self.model.classifier.add_module('Our Sigmoid', nn.Sigmoid())

    def forward(self, input1, input2):
        feat1 = self.features(input1)
        f1_avg = self.ap(feat1)
        f1_max = self.mp(feat1)
        f1 = torch.cat((f1_avg, f1_max), dim=1)
        f1 = self.fla(f1)
        # f1 = feat1.view(feat1.size()[0], -1)  # make it suitable for fc layer.
        # feat1 /= torch.sqrt(torch.sum(feat1**2, dim=1, keepdim=True))
        feat2 = self.features(input2)
        f2_avg = self.ap(feat2)
        f2_max = self.mp(feat2)
        f2 = torch.cat((f2_avg, f2_max), dim=1)
        f2 = self.fla(f2)
        # f2 = feat2.view(feat2.size()[0], -1)  # make it suitable for fc layer.
        # feat2 /= torch.sqrt(torch.sum(feat2 ** 2, dim=1, keepdim=True))
        # feat = feat1 + feat2

        # f1_max = torch.nn.functional.max_pool2d(feat1, kernel_size=feat1.size()[2:])
        # f1_avg = nn.functional.avg_pool2d(feat1, kernel_size=feat1.size()[2:])
        # f1 = torch.cat((f1_max, f1_avg), dim=1)
        #
        # f2_max = torch.nn.functional.max_pool2d(feat2, kernel_size=feat2.size()[2:])
        # f2_avg = nn.functional.avg_pool2d(feat2, kernel_size=feat2.size()[2:])
        # f2 = torch.cat((f2_max, f2_avg), dim=1)

        f3 = torch.sub(f1, f2)
        f3 = torch.mul(f3, f3)

        f1_ = torch.mul(f1, f1)
        f2_ = torch.mul(f2, f2)
        f4 = torch.sub(f1_, f2_)

        f5 = torch.mul(f1, f2)

        feat = torch.cat((f5, f4, f3), dim=1)

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
        count_params(self)

    def train(self):
        self.features.train()
        self.classifier.train()

    def eval(self):
        self.features.eval()
        self.classifier.eval()
