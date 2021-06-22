from torch.nn.modules.loss import _WeightedLoss

class LabelSmoothingCrossEntropy(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.1, ignore_index=-100):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
#             targets = torch.empty(size=(targets.size(0), n_classes),
#                                   device=targets.device) \
#                 .fill_(smoothing / (n_classes - 1)) \
#                 .scatter_(-1, targets.data.unsqueeze(-1), 1. - smoothing) # dim=-1로 수정
            
            ret_targets = torch.full((n_classes,), smoothing / (n_classes-2)) # size 만큼 smoothing_value로 채우기
            if self.ignore_index != -100:
                ret_targets[self.ignore_index] = 0.0 # pad
            ret_targests = targets.repeat(targets.size(0), 1)\
                            .scatter(1, targets.unsqueeze(1), 1 - smoothing)
        return ret_targests
    
            

    def forward(self, inputs, targets):
        targets = LabelSmoothingCrossEntropy._smooth_one_hot(targets, inputs.size(-1), self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss