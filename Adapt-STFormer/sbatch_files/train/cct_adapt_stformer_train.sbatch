#!/bin/bash


#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100|rtx6000"
#SBATCH --job-name=torch 
#SBATCH --account=torch_pr_51_general

module purge

singularity exec --nv \
    --overlay /scratch/yl9727/my_env/overlay-15GB-500K.ext3:ro \
    /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c \
    "source /ext3/env.sh; \
    python svpr_main.py \
		--arch adapt_stformer \
		--pooling seqvlad \
		--trunc_te 8 --freeze_te 1 \
		--nNeg 5 --seqL 5 \
		--batchSize 4 \
		--cacheBatchSize 16 \
		--optim ADAM \
		--lr 0.00001 \
		--nEpochs 200 \
		--descType raw_384 \
		--img_shape 384 384 \
		--outDims 24576 \
		--backbone cct384 \
		--dataset oxford-easy \
		--patience 5 \
		--k_points 8 \
        --num_levels 2 \
		--weightDecay 0 "