import torch.nn as nn
import torchvision.models as models


def create_resnet_feature_extractor(output_dim=512):
    resnet = models.resnet18(pretrained=True)
    layers = list(resnet.children())[:-1]
    feature_extractor = nn.Sequential(*layers)
    fc = nn.Linear(resnet.fc.in_features, output_dim)
    return feature_extractor, fc

class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, output_dim=512):
        super(ResNet18FeatureExtractor, self).__init__()
        self.feature_extractor, self.fc = create_resnet_feature_extractor(output_dim)
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

    