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
import torchvision
import torch
import torch.nn.functional as F

# from contrastiveloss_helperfunc import SupConLoss
from contrast_single_positive import SupConLoss

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
    horizontal_flip = torchvision.transforms.RandomHorizontalFlip(p=0.5)
    random_erasing = torchvision.transforms.RandomErasing(p=0.25)

    return_samples = []

    # Loop through each item in the batch:
    for i in range(samples.size(0)):
        fbank = samples[i,0]
        fbank = fbank.transpose(0, 1).unsqueeze(0)
        if freqm != 0:
            fbank = freqm_mask(fbank)
        if timem != 0:
            fbank = timem_mask(fbank)
        # Horizontal Flip:
        #fbank = horizontal_flip(fbank)
        # Random Erasing:
#        fbank = random_erasing(fbank)

        fbank = fbank.squeeze().transpose(0, 1)  # back to (1024, 128)
        fbank = (fbank - norm_mean) / (norm_std * 2)
        if noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)
        return_samples.append(fbank.unsqueeze(0))

    samples = torch.stack(return_samples, dim=0)
    return samples


def consistency_reg(a, b):
    sig_a = torch.sigmoid(a)
    sig_b = torch.sigmoid(b)

    eps = 1e-7
    sig_a_detach = sig_a.detach()
    a_b = -sig_a_detach * (torch.log(sig_b + eps)) - (1-sig_a_detach) * (torch.log(1-sig_b+eps))
    sig_b_detach = sig_b.detach()
    b_a = -sig_b_detach * (torch.log(sig_a + eps)) - (1-sig_b_detach) * (torch.log(1-sig_a+eps))

    loss = 0.5 * (a_b + b_a).mean()
    return loss

def consistency_reg_n(parts):
    count=0
    loss=0
    for i in range(len(parts)):
        for j in range(i+1,len(parts)):
            loss+= consistency_reg(parts[i],parts[j])
            count+=1
    print(f"Loss: {loss}, Count:{count}, Loss/Count: {loss/count}")
    return loss/count



def custom_loss_function(leaf_nodes, targets, features, temperature, device):
    batch_size, num_classes = targets.shape
    feat_dim = features.shape[2]
    labels_full = torch.arange(num_classes).unsqueeze(0).repeat(batch_size, 1).to(device)

    layer_loss = []
    # Later: take this out and place in main_finetune_as.py
    sup_con_loss = SupConLoss(temperature=temperature)
    max_depths = 5

    # TODO: make this configurable.
    features = F.normalize(features, dim=-1)

    for l in range(0, max_depths+1):
        layer = max_depths-l # We start from the bottom

        node_list = leaf_nodes[str(layer+1)]
        targets_layer = targets[:, node_list]
        labels_layer = labels_full[:, node_list]

        # exp(T * T^T)/temp where T=targets_layer

        # This is a 1D tensor, selecting representations of present events.
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
        current_layer_loss = sup_con_loss(features = feats_layer.unsqueeze(1), mask=mask_labels)

        print(f"Layer {layer} number of event instances: {nnz}")
        print(f"Layer {layer} number of event classes: {torch.unique(labs).shape[0]}")
        print(f"Layer {layer} number of positive pairs: {int(mask_labels.sum())}")

        print(f"Layer {layer} loss: {current_layer_loss}")

        # ADD A PENALTY LOSS DUE TO LAYER
        current_penalty = 1 #/(max_depths - layer +1)   #np.exp(1/((max_depths-layer)+1))
        layer_loss_penalty = current_layer_loss * current_penalty
        layer_loss.append(layer_loss_penalty)
         
    return sum(layer_loss)

def linear_cr(samples,targets,batch_size,audio_conf,args,model,criterion,scale1=0.8,scale2=0.2):
    # Generate Augmentation
    print(f"Linear CR activated with scalars: {scale1} and {scale2}")
    samples_with_aug = torch.cat([samples,samples])
    samples_with_aug = specAug(samples_with_aug,audio_conf)
    samples = samples_with_aug[:batch_size]
    aug = samples_with_aug[batch_size:]

    with torch.cuda.amp.autocast():
        outputs_samples, feats  = model(samples, mask_t_prob=args.mask_t_prob, mask_f_prob=args.mask_f_prob)
        bce_loss_samples = criterion(outputs_samples, targets)
        outputs_aug, feats  = model(aug, mask_t_prob=args.mask_t_prob, mask_f_prob=args.mask_f_prob)
        bce_loss_aug = criterion(outputs_aug, targets)
    output1 = scale1*outputs_samples + scale2*outputs_aug
    y = scale1*samples + scale2*aug
    with torch.cuda.amp.autocast():
        output2, feats  = model(y, mask_t_prob=args.mask_t_prob, mask_f_prob=args.mask_f_prob)
        bce_loss2 = criterion(output2, targets)

    return consistency_reg(output1,output2), bce_loss_samples, bce_loss_aug, bce_loss2

def correct_linear_cr(x1,targets, batch_size,audio_conf, model,criterion, args):
    mixup = np.random.beta(10,10)   #torch.from_numpy(np.random.beta(10,10))
    x2 = torch.roll(x1, dims=0,shifts=1)
    targets2 = torch.roll(targets,dims=0,shifts=1)
    targets3 = mixup*targets + (1-mixup)*targets2

#    print(f"Shape of x2: {x2.shape}, shape of mixup: {mixup.shape}")
    x3 = mixup*x1 + (1-mixup)*x2
    concat_data = torch.cat([x1,x2,x3])
    concat_data_aug = specAug(concat_data,audio_conf)
    targets = torch.cat([targets,targets2,targets3])
    with torch.cuda.amp.autocast():
        outputs, feats = model(concat_data_aug,mask_t_prob=args.mask_t_prob, mask_f_prob=args.mask_f_prob)
        bce_loss = criterion(outputs,targets)
    logits = torch.split(outputs,batch_size)
    logits_1 = logits[0]
    logits_2 = logits[1]
    logits_y3 = logits[2]
    logits_y3_hat = torch.log(torch.sigmoid(logits_1)*mixup + torch.sigmoid(logits_2)*(1-mixup))

    return consistency_reg(logits_y3, logits_y3_hat), bce_loss

#change targets



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
        print(f"samples.shape: {samples.shape}")

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if args.linear_cr and data_aug: # disable earlier mixup
            #consistency_reg_loss, bce_loss = linear_cr(samples,targets,samples.shape[0],audio_conf,args,model,criterion)
            consistency_reg_loss, bce_loss = correct_linear_cr(samples,targets,samples.shape[0],audio_conf, model,criterion,args)
            bce_loss = bce_loss
            consistency_constant = args.consistency_constant
        else:
            if data_aug:
# remove loop   
                samples = torch.cat([samples]*args.num_augs)
                targets = torch.cat([targets]*args.num_augs)
                samples = specAug(samples,audio_conf)
                print(f"Shape of samples {samples.shape} and target {targets.shape} have been augmented {args.num_augs} times.")

            # Get outputs from the ViT model and calculate loss
            with torch.cuda.amp.autocast():
                outputs, feats  = model(samples, mask_t_prob=args.mask_t_prob, mask_f_prob=args.mask_f_prob)
                bce_loss = criterion(outputs, targets)

            if data_aug and args.consistency_regularization:
                batch_size = outputs.shape[0] // args.num_augs
                parts = torch.split(outputs,batch_size)
                consistency_reg_loss = consistency_reg_n(parts)
                consistency_constant = args.consistency_constant
            else:
                consistency_reg_loss = 0
                consistency_constant = 0


        print(f"Consistency Reg Loss: {consistency_reg_loss}")
        print(f"BCE loss: {float(bce_loss)}")
        bce_cons_loss = bce_loss + (consistency_reg_loss * consistency_constant)
        print(f"BCE + Constiency_Regulatizaton: {bce_cons_loss}")


        if args.label_dep_classification and args.sup_con_loss_weight:
            assert feats is not None
            contrastive_loss = custom_loss_function(layer_leafs, targets, feats, args.sup_con_loss_temperature, device)
            print(f"Contrastive Loss: {float(contrastive_loss)}")
        else:
            contrastive_loss = 0.0

        # TODO: configure the coefficient.
        loss = bce_cons_loss + args.sup_con_loss_weight * contrastive_loss

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
