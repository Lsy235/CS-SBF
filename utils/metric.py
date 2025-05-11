import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss

class CompactnessLoss(nn.Module):
    def __init__(self, num_branch=9, feat_dim=128, use_gpu=True):
        super(CompactnessLoss, self).__init__()
        self.num_branch = num_branch
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(
                torch.randn(self.num_branch, self.feat_dim).cuda()
            )
        else:
            self.centers = nn.Parameter(torch.randn(self.num_branch, self.feat_dim))

    def forward(self, x):
        dist = (x - self.centers).pow(2).sum(dim=-1).sum(dim=-1)
        loss = torch.clamp(dist, min=1e-12, max=1e12).mean(dim=-1)
        return loss

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        assert 0 <= smoothing < 1
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, pred, target):
        with torch.no_grad():
            true_dist = torch.empty(size=(target.size(0), self.num_classes), device=target.device)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * F.log_softmax(pred, dim=-1), dim=-1))

class SPLlabelSmoothing(nn.Module):
    def __init__(self, num_classes, smoothing=0.1, thrSPL = 1.25, growing_factor = 1.12):
        super(SPLlabelSmoothing, self).__init__()
        assert 0 <= smoothing < 1
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.thrSPL = thrSPL
        self.growing_factor = growing_factor

    def forward(self, pred, target):
        with torch.no_grad():
            true_dist = torch.empty(size=(target.size(0), self.num_classes), device=target.device)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        superLoss = torch.sum(-true_dist * F.log_softmax(pred, dim=-1), dim=-1)
        # print(f"superLoss: {superLoss}")
        v = self.spl_loss(superLoss)
        return torch.mean(v * superLoss)

    def increase_threshold(self):
        self.thrSPL *= self.growing_factor

    def spl_loss(self, super_loss):
        # 如果 模型的loss < threshold --> v=1,表示该数据集简单
        # 否则                       --> v=0,表示该数据集难
        v = super_loss < self.thrSPL
        self.increase_threshold()
        return v.int()