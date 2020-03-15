import torch
from torch import nn
from torchvision import models


class SiameseNetwork(nn.Module):  # A simple implementation of siamese network, ResNet50 is used, and then connected by three fc layer.
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.feat_ext = models.vgg16(pretrained=True)
        for param in self.feat_ext.parameters():
            param.requires_grad = False

        num_features = self.feat_ext.classifier[-1].in_features
        print("features space size: {}".format(num_features))
        features = list(self.feat_ext.classifier.children())[:-1]  # Remove last layer
        self.feat_ext.classifier = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Linear(2*num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64,2))
            # ,nn.Sigmoid())

        self.initialize()

    def forward(self, input1, input2):
        feat1 = self.feat_ext(input1)
        feat1 = feat1.view(feat1.size()[0], -1)  # make it suitable for fc layer.
        feat2 = self.feat_ext(input2)
        feat2 = feat2.view(feat2.size()[0], -1)  # make it suitable for fc layer.
        output = torch.cat((feat1, feat2), 1)
        output = self.classifier(output)
        return output

    # init weights of our classifier
    def initialize(self):
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.zero_()
        self.classifier.apply(init_weights)
