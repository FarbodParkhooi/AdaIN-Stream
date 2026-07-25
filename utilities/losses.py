from utilities.configs import Configs
from types import SimpleNamespace
from torch import nn
import torchvision
import torch

configs = Configs()

weights = SimpleNamespace(
    content =  configs.content_weight,
    style   =  configs.style_weight
)

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
    std_s = torch.std(style_feat, dim=[2,3], keepdim=True, unbiased=False)   + 1e-5
    std_c = torch.std(content_feat, dim=[2,3], keepdim=True, unbiased=False) + 1e-5
    content_norm = (content_feat - mean_c) / std_c
    target = std_s * content_norm + mean_s
    
    return target

def gram_matrix(feat):
    F = torch.flatten(start_dim=2, end_dim=-1)
    C, H, W = feat.shape[1], feat.shape[2], feat.shape[3]
    G = F @ F.transpose(1,2)
    G_normalised = G / (C * H * W)

    return G_normalised

def content_loss(output_feats, content_feats, style_feats):
    out = output_feats["relu4_2"]
    cont = content_feats["relu4_2"]
    sty = style_feats["relu4_2"]
    target = adain_target(cont, sty)
    
    return torch.mean((out - target) ** 2)

def style_loss(output_feats, style_feats):
    total_loss = 0
    layers_name = ["relu1_1", "relu2_1", "relu3_1", "relu4_1"]
    for layer in layers_name:
        out_feat = output_feats[layer]
        sty_feat = style_feats[layer]
        G_out = gram_matrix(out_feat)
        G_sty = gram_matrix(sty_feat)
        loss_layer = torch.mean((G_out - G_sty) ** 2)
        total_loss += loss_layer
    
    return total_loss

def total_loss(output, content, style, weights):
    of, cf, sf = feature_extraction(output), feature_extraction(content), feature_extraction(style)
    L_content = content_loss(of, cf, sf)
    L_style = style_loss(of, sf)

    return weights.content * L_content + weights.style * L_style
