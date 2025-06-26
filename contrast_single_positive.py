import torch.nn as nn
import torch
import numpy as np

    
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    """
    Author: Yonglong Tian (yonglong@mit.edu)
    Date: May 07, 2020
    """
    def __init__(self, temperature=0.07, contrast_mode='all', device='cuda'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.device = device

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        # device = (torch.device(self.device)
        #           if features.is_cuda
        #           else torch.device('cpu'))
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            # TODO shouldn't be one
            labels = labels.contiguous().view(-1, 1) # [batch_size, 1]
            if labels.shape[0] != batch_size:
                raise ValueError(f'Num of labels does not match num of features, {labels.shape[0]} != {batch_size}')
            # compute the 2d mask
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # features: [bsz, n_views, f_dim]
        # mask.shape = [batch_size, batch_size]

        # PROBLEMS!! Not fixed label size
        contrast_count = features.shape[1]

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # [batch_size * n_views, feature_size]
        
        self.contrast_mode = 'all'

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        pos_mask = mask.repeat(anchor_count, contrast_count) # [batch_size * n_views, batch_size * n_views]
        neg_mask = 1.0 - pos_mask

        # mask-out self-contrast cases (the diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        pos_mask = pos_mask * logits_mask
        neg_mask = neg_mask * logits_mask

        eplison = 1e-8
        # compute log_prob
        exp_logits_pos = torch.exp(logits) * pos_mask # [batch_size * n_views, batch_size * n_views]
        exp_logits_neg = torch.exp(logits) * neg_mask # [batch_size * n_views, batch_size * n_views]
        # There is a single positive in the denominator.
        log_prob = logits - torch.log(exp_logits_pos + exp_logits_neg.sum(1, keepdim=True) + eplison) # [batch_size * n_views, batch_size * n_views]
        # compute mean of log-likelihood over positive
        loss = - (pos_mask * log_prob).sum() / (pos_mask.sum() + eplison)
        return loss
