import torch
import torch.nn as nn


class MixupLayer(nn.Module):
    def __init__(self, prob, alpha=1.0):
        super(MixupLayer, self).__init__()
        self.prob = prob

    def forward(self, X1, y1):
        if not self.training:
            return X1, y1
        lambda_ = torch.rand(X1.size(0), device=X1.device).view(-1, 1)
        X_l = lambda_
        y_l = lambda_.view(-1, 1)
        X2 = X1.flip(0) 
        X = X1 * X_l + X2 * (1 - X_l)
        y2 = y1.flip(0) 
        y = y1 * y_l + y2 * (1 - y_l)
        dec = torch.rand(X1.size(0), device=X1.device) < self.prob
        dec = dec.view(-1, 1).float()
        out1 = dec * X + (1 - dec) * X1
        out2 = dec.view(-1, 1) * y + (1 - dec.view(-1, 1)) * y1
        return out1, out2