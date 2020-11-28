import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import numpy as np
import cv2 as cv
import sys
import os


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i *
                                growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, in_channels=3):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_features,
                                kernel_size=5, stride=2, padding=2, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.5)

    def __init__(self, pth, mode="cpu"):
        super(DenseNet, self).__init__()
        self.device = torch.device(mode)
        self.DenseNet = torch.load(pth, map_location=torch.device(mode))

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = self.avg_pool(out).view(features.size(0), -1)
        out = self.drop(out)
        out = self.classifier(out)
        return out

    def pred(self, gray_version):
        self.DenseNet.eval()

        with torch.no_grad():
            resized_gray = gray_version.permute(
                2, 0, 1).unsqueeze(0).to(self.device)
            oh = self.DenseNet(resized_gray)

            pred_class = oh[:, :20].view(-1, 10).argmax(1).cpu().numpy()
            pred_box = oh[:, 20:].long().cpu().numpy()[0].reshape(2, 4)

        return pred_class, pred_box


def classify_and_detect(images):
    """
    :param np.ndarray images: N x 4096 array containing N 64x64 images flattened into vectors
    :return: np.ndarray, np.ndarray
    """
    N = images.shape[0]

    # pred_class: Your predicted labels for the 2 digits, shape [N, 2]
    pred_class = np.empty((N, 2), dtype=np.int32)
    # pred_bboxes: Your predicted bboxes for 2 digits, shape [N, 2, 4]
    pred_bboxes = np.empty((N, 2, 4), dtype=np.float64)

    # add your code here to fill in pred_class and pred_bboxes

    # model = DenseNet('/content/drive/MyDrive/Colab Notebooks/weights.pt', "cuda" if torch.cuda.is_available() else "cpu")

    # IF NOT RUNNING ON COLAB
    cwd = os.getcwd()
    weightsFile = cwd + '/weights.pt'
    model = DenseNet(
        weightsFile, "cuda" if torch.cuda.is_available() else "cpu")

    for i in range(N):
        gray = cv.cvtColor(images[i, :].reshape(64, 64), cv.COLOR_GRAY2BGR)
        # Ref: https://discuss.pytorch.org/t/convert-rgb-to-gray/49024
        gray_version = torch.from_numpy(gray.astype(np.float32)/255.)

        class_label, bbox = model.pred(gray_version)
        pred_class[i, :] = class_label
        pred_bboxes[i, :] = bbox

        if i % 50 == 0:
            print('{}/{}\n'.format(i, N))

    return pred_class, pred_bboxes
