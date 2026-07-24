from torch import nn
import torchvision
import torch

# Load the model
vgg_model = torchvision.models.vgg19(pretrained=True)
# Freezing the parameters
for param in vgg_model.parameters():
    param.requires_grad = False

slice_1 =  nn.Sequential(*list(vgg_model.features.children())[:2])
slice_2 =  nn.Sequential(*list(vgg_model.features.children())[2:7])
slice_3 =  nn.Sequential(*list(vgg_model.features.children())[7:12])
slice_4 =  nn.Sequential(*list(vgg_model.features.children())[12:21])
slice_5 =  nn.Sequential(*list(vgg_model.features.children())[21:23])

def feature_extraction(image_batch):
    relu_1_1_features = slice_1(image_batch)
    relu_2_1_features = slice_2(relu_1_1_features)
    relu_3_1_features = slice_3(relu_2_1_features)
    relu_4_1_features = slice_4(relu_3_1_features)
    relu_4_2_features = slice_5(relu_4_1_features)
    featurs = {
        "relu1_1" : relu_1_1_features,
        "relu2_1" : relu_2_1_features,
        "relu3_1" : relu_3_1_features,
        "relu4_1" : relu_4_1_features,
        "relu4_2" : relu_4_2_features
    }
    return featurs

def adain_target(content_feat, style_feat):
    mean = torch.mean(style_feat, dim=[2,3], keepdim=True)