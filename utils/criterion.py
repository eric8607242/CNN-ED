import torch
import torch.nn as nn

class Criterion(nn.Module):
    def __init__(self, alpha=0.1):
        super(Criterion, self).__init__()

        self.alpha = alpha

    def forward(self, anchor_outs, positive_outs, negative_outs, positive_distance, negative_distance):
        triplet_loss = self._triplet_forward(anchor_outs, positive_outs, negative_outs, positive_distance, negative_distance)
        approximation_loss = self._approximation_forward(anchor_outs, positive_outs, negative_outs, positive_distance, negative_distance)

        loss = triplet_loss + self.alpha * approximation_loss
        return loss, triplet_loss, approximation_loss

    def _triplet_forward(self, anchor_outs, positive_outs, negative_outs, positive_distance, negative_distance):
        mu = positive_distance - negative_distance
        postive_outs_distance = torch.norm(anchor_outs - positive_outs, dim=1)
        negative_outs_distance = torch.norm(anchor_outs - negative_outs, dim=1)

        loss = torch.max(0, postive_outs_distance - negative_outs_distance - mu)
        return loss

    def _approximation_forward(self, anchor_outs, positive_outs, negative_outs, positive_distance, negative_distance):
        postive_outs_distance = torch.norm(anchor_outs - positive_outs, dim=1)
        negative_outs_distance = torch.norm(anchor_outs - negative_outs, dim=1)

        positive_approximation = torch.norm(postive_outs_distance - positive_distance, dim=1)
        negative_approximation = torch.norm(negative_outs_distance - negative_distance, dim=1)

        loss = positive_approximation + negative_approximation
        return loss


