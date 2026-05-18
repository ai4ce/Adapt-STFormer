#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=25:00:00
#SBATCH --mem=150GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=torch 
#SBATCH --constraint="a100|rtx8000"
#SBATCH --account=pr_110_tandon_priority

module purge

singularity exec --nv \
    --overlay /scratch/yl9727/my_env/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c \
    "source /ext3/env.sh; \
    python svpr_main.py \
	--arch stformer \
	--trunc_te 4 --freeze_te -1 \
	--trunc_te_tatt 4 --freeze_te_tatt -1 \
	--nNeg 5 --seqL 5 \
	--batchSize 4 \
	--cacheBatchSize 16 \
	--optim ADAM \
	--nEpochs 100 \
	--descType raw_384 \
	--img_shape 384 384 \
	--outDims 24576 \
	--backbone cct384 \
	--rel_pos_temporal \
	--abs_pos_embed \
	--patience 5 \
	--dataset oxford-hard \
	--ckpt external_pretrain \
	--weightDecay 0 \
	--resume /scratch/yl9727/adaptiveNet/AdaptiveNet/checkpoints/ours/msls.pth"