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
    mean_s = torch.mean(style_feat, dim=[2,3], keepdim=True)
    mean_c = torch.mean(content_feat, dim=[2,3], keepdim=True)
    std_s = torch.std(style_feat, dim=[2,3], keepdim=True, unbiased=False)
    std_c = torch.std(content_feat, dim=[2,3], keepdim=True, unbiased=False)
    content_norm = (content_feat - mean_c) / std_c
    target = std_s * content_norm + mean_s
    return target

def gram_matrix(feat):
    F = torch.flatten(start_dim=2, end_dim=-1)
    G = F @ F.transpose(1,2)
    return G

def content_loss(output_feats, content_feats, style_feats):
    of_ext = feature_extraction(output_feats)["relu4_2"]
    cf_ext = feature_extraction(content_feats)["relu4_2"]
    sf_ext = feature_extraction(style_feats)["relu4_2"]
