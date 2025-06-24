# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional
import numpy as np
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

from contrastiveloss_helperfunc import SupConLoss
import json
from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
from util.stat import calculate_stats, concat_all_gather

def custom_loss_function(label_depths, leaf_nodes, labels, features, device):
    layer_loss = []
    # Later: take this out and place in main_finetune_as.py
    sup_con_loss = SupConLoss(temperature=0.1, base_temperature=0.1)
    max_depths = 5

    features = F.normalize(features, dim=-1)

    for l in range(0, max_depths+1):
        layer = max_depths-l # We start from the bottom

        # Iterate through each leaf
        leaf_loss = []
        x = []
        for k in leaf_nodes[str(layer+1)]:

            labels_k = labels[:, k]
            # mask_ij is True, if both i and j have label 1.
            mask_labels = (labels_k[:, None] == 1) & (labels_k[None, :] == 1)
            # remove self-positive pairs.
            mask_diagonal = torch.eye(*mask_labels.shape, dtype=torch.bool, device=labels.device)
            mask_labels.masked_fill_(mask_diagonal, 0)
            # Skip classes without data.
            if mask_labels.sum() == 0:
                continue

            x.append(mask_labels.sum().item())
 #           print(labels[0])
 #           print(mask_labels[0])
            mask_labels=mask_labels.to(device)
            #print(f"Nonzero elements in mask_labels: {mask_labels.sum()}")

            sliced_feature = features[:, k:k+1, :]
            sliced_feature = sliced_feature.to(device)

            # Calculate Leaf Loss
            # print(f"Mask Label Sum: {mask_labels.sum()}")
            layer_leaf_loss = sup_con_loss(features = sliced_feature, mask=mask_labels).to(device)
            #layer_leaf_loss.to(device)

            leaf_loss.append(layer_leaf_loss)

        print(f"Layer {layer} mask_labels sum: {sum(x)/len(x)}")
        # ADD A PENALTY LOSS DUE TO LAYER
        current_layer_loss = sum(leaf_loss)/len(leaf_loss)
        current_penalty = 1 #/(max_depths - layer +1)   #np.exp(1/((max_depths-layer)+1))
        layer_loss_penalty = current_layer_loss*current_penalty
        layer_loss.append(layer_loss_penalty)
         
    return sum(layer_loss)




def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,label_depths,layer_leafs,max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 500

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets, _vids) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

#        print(f"Shape of targets: {targets.shape}")

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Get Label Embeddings for batch
        batch_size = targets.shape[0]
        # shape [batch, num_classes, dim]
        label_embeddings = model.embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        # Get outputs from the ViT model and calculate loss
        # with torch.cuda.amp.autocast():
        #     outputs, _  = model(samples, mask_t_prob=args.mask_t_prob, mask_f_prob=args.mask_f_prob)
        #     loss = criterion(outputs, targets)

        # No length reduction when mask probs are 0.
        _, feats  = model(samples, mask_t_prob=0.0, mask_f_prob=0.0)

        # Perform Cross Attention between Label Embeddings and masked audio samples
        attn_output, attn_output_weights = model.multihead_attn(label_embeddings, feats, feats)

        attn_output = attn_output.to(device)#
        attn_output_weights = attn_output_weights.to(device)#

        # Alternative way to get outputs.
        outputs = (label_embeddings * attn_output).sum(-1)
        loss =  criterion(outputs, targets)

        print(f"Attention Output Shape: {attn_output.shape}")#

        contrastive_loss = custom_loss_function(label_depths, layer_leafs, targets, attn_output, device)####

        print(contrastive_loss.requires_grad)
        print(contrastive_loss.grad_fn)
        print(f"Contrastive Loss: {contrastive_loss}")#
        print(f"BCE loss: {loss}")
        print(f"Loss Item: {loss.item()}")

        constant =1#

        loss = constant * contrastive_loss #+ loss

        print(f"New loss: {loss}\n")

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

#        if model.embedding.weight.grad is not None:
#            print("Contrastive grad norm:", model.embedding.weight.grad.norm())
#        else:
#            print("Contrastive grad is None!")



        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:

            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
        #print("label_embeddings.requires_grad:", label_embeddings.requires_grad)
        #print("attn_output.requires_grad:", attn_output.requires_grad)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, dist_eval=False):
    criterion = torch.nn.BCEWithLogitsLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    outputs=[]
    targets=[]
    vids=[]
    for batch in metric_logger.log_every(data_loader, 300, header):

        images = batch[0]
        target = batch[1]
        vid = batch[2]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output, x_tran = model(images)
            # remark: 
            # 1. use concat_all_gather and --dist_eval for faster eval by distributed load over gpus
            # 2. otherwise comment concat_all_gather and remove --dist_eval one every gpu
            #if dist_eval:
            #    output = concat_all_gather(output)
            #    target = concat_all_gather(target)
            outputs.append(output)
            targets.append(target)
            vids.append(vid)

    outputs=torch.cat(outputs).cpu().numpy()
    targets=torch.cat(targets).cpu().numpy()
    vids = [j for sub in vids for j in sub]
    np.save('inf_output.npy', {'vids':vids, 'embs_527':outputs, 'targets':targets})
    stats = calculate_stats(outputs, targets)

    AP = [stat['AP'] for stat in stats]
    mAP = np.mean([stat['AP'] for stat in stats])
    print("mAP: {:.6f}".format(mAP))
    return {"mAP": mAP, "AP": AP}


