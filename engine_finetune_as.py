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




def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, layer_leafs,audio_conf,data_aug ,max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None, large_data_loader: Iterable = None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 500

    accum_iter = args.accum_iter
    optimizer.zero_grad()
    
    if large_data_loader is not None:
        large_iter = iter(large_data_loader)
        print("Large Data Loader exists...")
    

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets, _vids) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        print(f"samples.shape: {samples.shape}")

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if data_aug: 
            samples = torch.cat([samples]*args.num_augs)
            targets = torch.cat([targets]*args.num_augs)
            samples = specAug(samples,audio_conf)
            print(f"Shape of samples {samples.shape} and target {targets.shape} have been augmented {args.num_augs} times.")

        if large_data_loader is not None:
            try:
                large_sample, _,_ = next(large_iter)
            except StopIteration:
                large_iter = iter(large_data_loader)
                large_sample, _,_ = next(large_iter)

            large_samples = torch.cat([large_sample]*2)
            large_samples = specAug(large_samples,audio_conf).to(device, non_blocking=True)
            
            total_samples = torch.cat([large_samples,samples],dim=0) #[batch_size1+batch_size2, 1, 1024, 128]
        
        
        

        # Get outputs from the ViT model and calculate loss
        with torch.cuda.amp.autocast():
            if large_data_loader is not None:
                total_outputs,_  = model(total_samples, mask_t_prob=args.mask_t_prob, mask_f_prob=args.mask_f_prob)
                large_outputs = total_outputs[:large_samples.shape[0]]
                outputs = total_outputs[large_samples.shape[0]:]
                print(f"Shape of large outputs: {large_outputs.shape} and shape of outputs: {outputs.shape}")
            else:
                outputs,_  = model(samples, mask_t_prob=args.mask_t_prob, mask_f_prob=args.mask_f_prob)
            bce_loss = criterion(outputs, targets)

        large_batch_size = large_outputs.shape[0] // 2
        large_parts = torch.split(large_outputs,large_batch_size)
        large_consistency_reg_loss = consistency_reg_n(large_parts)
        print(f"Unlabeled Set Reg Loss: {large_consistency_reg_loss}")


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
        loss = bce_loss + (consistency_reg_loss * consistency_constant)+large_consistency_reg_loss
        print(f"BCE + Constiency_Regulatizaton: {loss}")

        contrastive_loss=0

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
