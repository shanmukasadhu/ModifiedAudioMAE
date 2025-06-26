# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import torchaudio
import math
import sys
from typing import Iterable, Optional
import numpy as np

import torch
import torch.nn.functional as F

from contrastiveloss_helperfunc import SupConLoss
from timm.data import Mixup

import util.misc as misc
import util.lr_sched as lr_sched
from util.stat import calculate_stats, concat_all_gather

def specAug(samples, audio_conf):
    freqm = audio_conf.get('freqm')
    timem = audio_conf.get('timem')
    norm_mean = audio_conf.get('mean')
    norm_std = audio_conf.get('std')
    freqm_mask = torchaudio.transforms.FrequencyMasking(freqm)
    timem_mask = torchaudio.transforms.TimeMasking(timem)
    noise = audio_conf.get('noise')


    return_samples = []

    # Loop through each item in the batch:
    for i in range(samples.size(0)):
        fbank = samples[i,0]
        fbank = fbank.transpose(0, 1).unsqueeze(0)
        if freqm != 0:
            fbank = freqm_mask(fbank)
        if timem != 0:
            fbank = timem_mask(fbank)
        fbank = fbank.squeeze().transpose(0, 1)  # back to (1024, 128)
        fbank = (fbank - norm_mean) / (norm_std * 2)
        if noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)
        return_samples.append(fbank.unsqueeze(0))

    samples = torch.stack(return_samples, dim=0)
    return samples


def custom_loss_function(leaf_nodes, targets, features, device):
    batch_size, num_classes = targets.shape
    feat_dim = features.shape[2]
    labels_full = torch.arange(num_classes).unsqueeze(0).repeat(batch_size, 1).to(device)

    layer_loss = []
    # Later: take this out and place in main_finetune_as.py
    sup_con_loss = SupConLoss(temperature=0.05)
    max_depths = 5

    # TODO: make this configurable.
    features = F.normalize(features, dim=-1)

    for l in range(0, max_depths+1):
        layer = max_depths-l # We start from the bottom

        node_list = leaf_nodes[str(layer+1)]
        targets_layer = targets[:, node_list]
        labels_layer = labels_full[:, node_list]

        # This is a 1D tensor.
        labs = torch.masked_select(labels_layer, targets_layer==1)
        nnz = labs.shape[0]
        feats_layer = torch.masked_select(features[:, node_list, :], (targets_layer==1).unsqueeze(-1)).reshape(nnz, feat_dim)

        # mask_ij is True, if i and j have the same labels.
        mask_labels = labs.unsqueeze(1) == labs.unsqueeze(0)
        # remove self-positive pairs.
        mask_diagonal = torch.eye(*mask_labels.shape, dtype=torch.bool, device=device)
        mask_labels.masked_fill_(mask_diagonal, 0)

        if mask_labels.sum() == 0:
            continue

        # Calculate Leaf Loss
        current_layer_loss = sup_con_loss(features = feats_layer, mask=mask_labels)

        leaf_loss.append(current_layer_loss)

        print(f"Layer {layer} number of event instances: {nnz}")
        print(f"Layer {layer} number of event classes: {torch.unique(labs).shape[0]}")
        print(f"Layer {layer} number of positive pairs: {int(mask_labels.sum())}")

        print(f"Layer {layer} loss: {current_layer_loss}")

        # ADD A PENALTY LOSS DUE TO LAYER
        current_penalty = 1 #/(max_depths - layer +1)   #np.exp(1/((max_depths-layer)+1))
        layer_loss_penalty = current_layer_loss * current_penalty
        layer_loss.append(layer_loss_penalty)
         
    return sum(layer_loss)




def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, layer_leafs,audio_conf,data_aug ,max_norm: float = 0,
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

        # Perform 2x Augmentations:
        if data_aug:
            samples = specAug(torch.cat([samples, samples], dim = 0),audio_conf)
            targets = torch.cat([targets, targets], dim = 0)
            print(f"Shape of samples {samples.shape} and target {targets.shape}  doubled")


        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Get outputs from the ViT model and calculate loss
        with torch.cuda.amp.autocast():
            outputs, feats  = model(samples, mask_t_prob=args.mask_t_prob, mask_f_prob=args.mask_f_prob)
            bce_loss = criterion(outputs, targets)
        print(f"BCE loss: {float(bce_loss)}")

        if args.label_dep_classification and args.sup_con_loss_weight:
            assert feats is not None
            contrastive_loss = custom_loss_function(layer_leafs, targets, feats, device)####
            print(f"Contrastive Loss: {float(contrastive_loss)}")#
        else:
            contrastive_loss = 0.0

        # TODO: configure the coefficient.
        loss = bce_loss + args.sup_con_loss_weight * contrastive_loss

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        print(f"New loss: {loss_value}\n")

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

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

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, float(contrastive_loss), float(bce_loss)


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
            # Calculate BCE Loss
            bce_loss = criterion(output, target)
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
    return {"mAP": mAP, "AP": AP}, float(bce_loss)
