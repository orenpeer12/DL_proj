import torch
from torch import nn
from torchvision import models
import numpy as np
import torch.functional as F

class SiameseNetwork(nn.Module):  # A simple implementation of siamese network, ResNet50 is used, and then connected by three fc layer.

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.feat_ext = models.vgg16_bn(pretrained=True)
        # self.feat_ext = models.squeezenet1_0(pretrained=True)
        for param in self.feat_ext.parameters():
            param.requires_grad = False

        num_features = self.feat_ext.classifier[6].in_features # VGG
        print("features space size: {}".format(num_features)) # VGG
        features = list(self.feat_ext.classifier.children())[:-1]  # Remove last layer in VGG16_bn
        # features = list(self.feat_ext.classifier.children())[:-1]  # Remove last layer in SN
        self.feat_ext.classifier = nn.Sequential(*features)

        # list(self.feat_ext.parameters())[-2].requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

        self.initialize()

    def forward(self, input1, input2):
        feat1 = self.feat_ext(input1)
        feat1 = feat1.view(feat1.size()[0], -1)  # make it suitable for fc layer.
        feat2 = self.feat_ext(input2)
        feat2 = feat2.view(feat2.size()[0], -1)  # make it suitable for fc layer.
        feat = feat1 + feat2
        output = self.classifier(feat)
        return output

    # init weights of our classifier
    def initialize(self):
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.zero_()
        self.classifier.apply(init_weights)
        trainable_model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        non_trainable_model_parameters = filter(lambda p: not(p.requires_grad), self.parameters())
        trainable_params = sum([np.prod(p.size()) for p in trainable_model_parameters])
        non_trainable_params = sum([np.prod(p.size()) for p in non_trainable_model_parameters])
        print("Num. of trainable parameters: {:,}, num. of frozen parameters: {:,}, total: {:,}".format(
            trainable_params, non_trainable_params, trainable_params + non_trainable_params))

    def train(self):
        self.feat_ext.eval()
        self.classifier.train()

    def eval(self):
        self.feat_ext.eval()
        self.classifier.eval()

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive