import torch
from torch import nn

from torchvision.utils import make_grid


class PatnetCNNModel(nn.Module):
    def __init__(self, args=None, **kwargs):
        super(PatnetCNNModel, self).__init__()
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.pretrained = dict_args.get("pretrained", None) # parameter and default value
        self.baseModel = dict_args.get("baseModel", None) # parameter and default value
        self.classes = dict_args.get("classes", 10) # parameter and default value
        self.featureSize = dict_args.get("featureSize", 2048) # parameter and default value
        self.net = None
        self.classifier = None

        if(self.baseModel == "vit_b_16"): # to load respective model as given in parameter
            from torchvision.models import vit_b_16
            model = vit_b_16(pretrained=self.pretrained)
            self.net = torch.nn.Sequential(*(list(model.children())[:-2]))
            self.classifier = torch.nn.Linear(self.featureSize, self.classes)
        if(self.baseModel == "efficientnet_v2_m"):
            from torchvision.models import efficientnet_v2_m
            model = efficientnet_v2_m(pretrained=self.pretrained)
            self.net = torch.nn.Sequential(*(list(model.children())[:-1]))
            self.classifier = torch.nn.Linear(self.featureSize, self.classes)
        if(self.baseModel == "regnet_y_16gf"):
            from torchvision.models import regnet_y_16gf
            model = regnet_y_16gf(weights="IMAGENET1K_V2")
            self.net = torch.nn.Sequential(*(list(model.children())[:-1]))
            self.classifier = torch.nn.Linear(self.featureSize, self.classes)
        if(self.baseModel == "resnext101_64x4d"):
            from torchvision.models import resnext101_64x4d
            model = resnext101_64x4d(weights="IMAGENET1K_V1")
            self.net = torch.nn.Sequential(*(list(model.children())[:-1]))
            self.classifier = torch.nn.Linear(self.featureSize, self.classes)
        if(self.baseModel == "resnet50_v1"):
            from torchvision.models import resnet50
            model = resnet50(weights="IMAGENET1K_V2")
            self.net = torch.nn.Sequential(*(list(model.children())[:-1]))
            self.classifier = torch.nn.Linear(self.featureSize, self.classes)
        if(self.baseModel == "resnet50_v2"):
            from torchvision.models import resnet50
            model = resnet50(weights="IMAGENET1K_V2")
            self.net = torch.nn.Sequential(*(list(model.children())[:-1]))
            self.classifier = torch.nn.Linear(self.featureSize, self.classes)
        if(self.baseModel == "resnet18"):
            from torchvision.models import resnet18
            model = resnet18(pretrained=self.pretrained)
            self.net = torch.nn.Sequential(*(list(model.children())[:-1]))
            self.classifier = torch.nn.Linear(self.featureSize, self.classes)
        if(self.baseModel == "resnet101"):
            from torchvision.models import resnet101
            model = resnet101(pretrained=self.pretrained)
            self.net = torch.nn.Sequential(*(list(model.children())[:-1]))
            self.classifier = torch.nn.Linear(self.featureSize, self.classes)
        if(self.baseModel == "vgg16"):
            from torchvision.models import vgg16
            model = vgg16(pretrained=self.pretrained)
            self.net = torch.nn.Sequential(*(list(model.children())[:-1]))
            self.classifier = torch.nn.Linear(self.featureSize, self.classes)
        if(self.baseModel == "vgg19"):
            from torchvision.models import vgg19
            model = vgg19(pretrained=self.pretrained)
            self.net = torch.nn.Sequential(*(list(model.children())[:-1]))
            self.classifier = torch.nn.Linear(self.featureSize, self.classes)
    def forward(self, x):
        x = self.net(x)
        return self.classifier(torch.squeeze(x))

    