import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SCAdaCos(nn.Module):
    def __init__(self, n_classes=10, n_subclusters=1, eps=1e-7, trainable=False, return_logit=False, emb_size=256):
        super(SCAdaCos, self).__init__()
        self.n_classes = n_classes
        self.n_subclusters = n_subclusters
        self.s_init = np.sqrt(2) * np.log(n_classes * n_subclusters - 1)
        self.eps = eps
        self.return_logit = return_logit
        # Weight initialization
        self.W = nn.Parameter(
            torch.Tensor(emb_size, n_classes * n_subclusters), requires_grad=trainable
        )
        nn.init.xavier_uniform_(self.W.data)

        # Scale factor
        self.s = nn.Parameter(torch.tensor(self.s_init), requires_grad=False)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y1):
        y1_orig = y1.clone()
        y1 = y1.repeat(1, self.n_subclusters)
        x = F.normalize(x, p=2, dim=1)
        # Normalize weights
        W = F.normalize(self.W, p=2, dim=0)
        # Dot product
        logits = torch.mm(x, W)
        # theta = torch.acos(torch.clamp(logits, -1.0 + self.eps, 1.0 - self.eps))
        # if self.training:
        #     with torch.no_grad():
        #         max_s_logits = torch.max(self.s * logits)
        #         B_avg = torch.exp(self.s * logits - max_s_logits)
        #         B_avg = torch.mean(torch.sum(B_avg, dim=1))
        #         theta_class = torch.sum(y1 * theta, dim=1)
        #         theta_med = torch.median(theta_class)
        #         self.s.data = (max_s_logits + torch.log(B_avg)) / torch.cos(
        #             min(torch.tensor(np.pi / 4), theta_med)
        #         )

        logits *= self.s
        out = F.softmax(logits, dim=1)
        out = out.view(-1, self.n_classes, self.n_subclusters)
        out = torch.sum(out, dim=2)
        loss = self.loss_fn(torch.log(out), y1_orig)
        if self.return_logit:
            return loss, out
        return loss
