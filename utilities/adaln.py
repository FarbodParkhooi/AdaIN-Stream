import torch

def AdalN(x, style:tuple):
    """
        style = (gamma, beta)
    """
    std = torch.std(x, dim=[2,3], keepdim=True, unbiased=False)
    mean = torch.mean(x, dim=[2, 3], keepdim=True)
    eps = 1e-5
    gamma, beta = style[0].unsqueeze(-1).unsqueeze(-1), style[1].unsqueeze(-1).unsqueeze(-1)
    x_norm = (x - mean) / (std + eps)
    x_styled = gamma * x_norm + beta
    return x_styled
