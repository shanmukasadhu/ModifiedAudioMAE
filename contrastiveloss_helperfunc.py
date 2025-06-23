import json
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class CrossAttention(nn.Module):
    def __init__(self, 
                q_dim: int,
                kv_dim: int,
                dim_head: int = 64, 
                num_heads: int = 8,
                qkv_bias: bool = False,
                qk_norm: bool = False,
                attn_drop: float = 0.,
                norm_layer: nn.Module = nn.LayerNorm,        
    ):
        super().__init__()
        self.inner_dim = dim_head * num_heads
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(q_dim, self.inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(kv_dim, self.inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(kv_dim, self.inner_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.to_out = nn.Linear(self.inner_dim, kv_dim, bias=False)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        B, N_kv, _ = x_kv.shape
        q = self.to_q(x_q).reshape(B, len(x_q[0]), self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        k = self.to_k(x_kv).reshape(B, N_kv, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        v = self.to_v(x_kv).reshape(B, N_kv, self.num_heads, self.dim_head).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, len(x_q[0]), self.inner_dim)
        x = self.to_out(x)
        return x



def unique(x, dim=None):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    unique, inverse = torch.unique(
        x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return unique, inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)


class HMLC(nn.Module):
    def __init__(self, temperature=0.07,
                 base_temperature=0.07, layer_penalty=None, loss_type='hmce', label_depths=None, device='cuda'):
        super(HMLC, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.device = device
        self.label_depths = label_depths # shape (num_labels, 1)
        if not layer_penalty:
            self.layer_penalty = self.pow_2
        else:
            self.layer_penalty = layer_penalty
        self.sup_con_loss = SupConLoss(temperature, device=device)
        self.loss_type = loss_type

    def pow_2(self, value):
        return torch.pow(2, value)

    def forward(self, features, labels):
        """
        :param features: shape (batch_size, num_labels, feature_dim)
        :param labels: shape (batch_size, num_labels)
        """
        device = (torch.device(self.device)
                   if features.is_cuda
                   else torch.device('cpu'))
        device = labels.device
        self.label_depths = self.label_depths.to(device)
        
        # mask = torch.ones(labels.shape).to(device) # shape (batch_size, 4 <- hierarchy)
        mask = labels.clone().detach().to(device)
        cumulative_loss = torch.tensor(0.0).to(device)
        max_loss_lower_layer = torch.tensor(float('-inf'))
        max_depths = int(torch.max(self.label_depths).item()) + 1
        # capture the loss for each layer by create a list of losses
        loss_by_depths = []
        mask_by_depths = []
        for l in range(1, max_depths):
            print(f"Layer: {max_depths-l}")
            # get depth_mask with those smaller than max_depths - l
            depth_mask = self.label_depths <= (max_depths - l) # shape (num_labels, 1)
            # check the device for depth_mask and mask
            assert depth_mask.device == mask.device, f'depth_mask.device: {depth_mask.device}, mask.device: {mask.device}'
            # filter the mask with depth_mask
            mask = mask * depth_mask # shape (batch_size, num_labels)
            # mask[:, labels.shape[1]-l:] = 0
            layer_labels = labels * mask

            #print(layer_labels.shape)
            print(f"Layer Labels: {layer_labels}")
            mask_labels = torch.stack([torch.all(torch.eq(layer_labels[i], layer_labels), dim=1)
                                       for i in range(layer_labels.shape[0])]).type(torch.uint8).to(device)
            print(" ")
            print(f"Mask Labels: {mask_labels}")
            print(" ")
            layer_loss = torch.Tensor(0) # self.sup_con_loss(features, mask=mask_labels)
            mask_by_depths.insert(0, mask_labels.detach().cpu().numpy())

            # l = l+1
            if self.loss_type == 'hmc':
                cumulative_loss += self.layer_penalty(torch.tensor(
                  l).type(torch.float)) * layer_loss
            elif self.loss_type == 'hce':
                layer_loss = torch.max(max_loss_lower_layer.to(layer_loss.device), layer_loss)
                cumulative_loss += layer_loss
            elif self.loss_type == 'hmce':
                layer_loss = torch.max(max_loss_lower_layer.to(layer_loss.device), layer_loss)
                tmp = torch.tensor(1 / (l)).type(torch.float).to(layer_loss.device)
                layer_loss = self.layer_penalty(tmp) * layer_loss
                cumulative_loss += layer_loss
                loss_by_depths.insert(0, layer_loss.detach().cpu().numpy()
                )
            else:
                raise NotImplementedError('Unknown loss')
            _, unique_indices = unique(layer_labels, dim=0)
            # max_loss_lower_layer = torch.max(
            #     max_loss_lower_layer.to(layer_loss.device), layer_loss)
            labels = labels[unique_indices]
            mask = mask[unique_indices]
            features = features[unique_indices]
        # return cumulative_loss / labels.shape[1], loss_by_depths, mask_by_depths
        # print(loss_by_depths[0].shape, mask_by_depths[0].shape)

        return cumulative_loss / labels.shape[1], np.array(loss_by_depths) / labels.shape[1]

    
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    """
    Author: Yonglong Tian (yonglong@mit.edu)
    Date: May 07, 2020
    """
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, device='cuda'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
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
        mask = mask.repeat(anchor_count, contrast_count) # [batch_size * n_views, batch_size * n_views]

        # mask-out self-contrast cases (the diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        eplison = 1e-8
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask # [batch_size * n_views, batch_size * n_views]
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + eplison) # [batch_size * n_views, batch_size * n_views]
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + eplison) # [batch_size * n_views]
        # print(mean_log_prob_pos)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        # print(loss)
        return loss
