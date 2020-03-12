import torch
from torch import nn
from torchvision import models


class SiameseNetwork(nn.Module):  # A simple implementation of siamese network, ResNet50 is used, and then connected by three fc layer.
    def __init__(self):
        class Identity(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x

        super(SiameseNetwork, self).__init__()
        # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
        # self.cnn1 = models.resnet50(pretrained=True) #resnet50 doesn't work, might because pretrained model recognize all faces as the same.
        # for param in self.cnn1.parameters():
        #     param.requires_grad = False
        # self.feat_ext = nn.Sequential(
        #     nn.ReflectionPad2d(1),
        #     nn.Conv2d(3, 64, kernel_size=3),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(64),
        #     nn.Dropout2d(p=.2),
        #
        #     nn.ReflectionPad2d(1),
        #     nn.Conv2d(64, 64, kernel_size=3),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(64),
        #     nn.Dropout2d(p=.2),
        #
        #     nn.ReflectionPad2d(1),
        #     nn.Conv2d(64, 32, kernel_size=3),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(32),
        #     nn.Dropout2d(p=.2),
        # )
        ###
        # self.feat_ext = models.resnet50(pretrained=True).to(device)
        ############# ->
        self.feat_ext = models.vgg16(pretrained=True)
        for param in self.feat_ext.parameters():
            param.requires_grad = False

        self.feat_ext.classifier[-1] = Identity()

        self.classifier = nn.Sequential(
            nn.Linear(2*4096, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid())

        ############# <-
        # self.fc1 = nn.Linear(2*1000, 500)
        # self.fc2 = nn.Linear(500, 256)
        # self.fc3 = nn.Linear(256, 1)

    def forward(self, input1, input2):  # did not know how to let two resnet share the same param.
        feat1 = self.feat_ext(input1)
        feat1 = feat1.view(feat1.size()[0], -1)  # make it suitable for fc layer.
        feat2 = self.feat_ext(input2)
        feat2 = feat2.view(feat2.size()[0], -1)  # make it suitable for fc layer.
        output = torch.cat((feat1, feat2), 1)
        output = self.classifier(output)

        return output