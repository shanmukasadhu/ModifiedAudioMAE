#!/bin/bash
#SBATCH --job-name=aud
#SBATCH --partition=learnlab
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH --mem=240GB
#SBATCH --signal=USR1@120
#SBATCH --constraint=volta32gb

#SBATCH --output=/checkpoint/ssadhu/jobs/%A.out
#SBATCH --error=/checkpoint/ssadhu/jobs/%A.err

if [ -z "$1" ]
then
	blr=1e-3
else
	blr=$1
fi

if [ -z "$2" ]
then
	ckpt=ckpt/pretrained.pth
else
	ckpt=$2
fi


audioset_train_json=/checkpoint/berniehuang/ast/egs/audioset/data/datafiles/train.json
audioset_train_all_json=/checkpoint/berniehuang/ast/egs/audioset/data/datafiles/train_all.json
audioset_eval_json=/checkpoint/berniehuang/ast/egs/audioset/data/datafiles/eval_19k.json
audioset_label=/checkpoint/berniehuang/ast/egs/audioset/data/class_labels_indices.csv



audioset_train_json=/checkpoint/berniehuang/ast/egs/audioset/data/datafiles/train.json
audioset_train_all_json=/home/ssadhu/EquiAV/arkjobs/filtered_metadata3579.json


audioset_eval_json=/data/not_backed_up/ssadhu/ark_files/test_ark_set_with_offsets.json

audioset_label=/home/ssadhu/EquiAV/datasets/dataprep/AudioSet_2M/class_labels_indices.csv


audioset_train_weight=/checkpoint/berniehuang/mae/data/audioset/weight_train.csv
audioset_train_all_weight=/home/ssadhu/EquiAV/datasets/dataprep/AudioSet_2M/weights.csv


dataset=audioset


CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main_finetune_as.py \
--model vit_base_patch16 \
--dataset $dataset \
--data_train $audioset_train_json \
--data_eval $audioset_eval_json \
--label_csv $audioset_label \
--finetune $ckpt \
--roll_mag_aug True \
--epochs 60 \
--blr $blr \
--batch_size 8 \
--warmup_epochs 4 \
--first_eval_ep 15 \
--dist_eval \
--mask_2d True \
--mask_t_prob 0.2 \
--mask_f_prob 0.2 \
