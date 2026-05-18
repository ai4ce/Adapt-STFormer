#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=01:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=torch 
#SBATCH --constraint="rtx8000|a100"
#SBATCH --account=pr_130_tandon_advanced

module purge

singularity exec --nv \
    --overlay /scratch/yl9727/my_env/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c \
    "source /ext3/env.sh; \
		python svpr_main.py \
		--mode test \
		--split test \
		--arch seqvlad \
		--pooling seqvlad \
		--trunc_te 8 --freeze_te 1 \
		--nNeg 5 --seqL 5 \
		--cacheBatchSize 16 \
		--descType raw_384 \
		--img_shape 384 384 \
		--outDims 24576 \
		--backbone cct384 \
		--dataset oxford-hard \
		--ckpt best \
		--weightDecay 0 \
		--pca_dim 4096 \
		--resume data/runs/seqvlad_cct384_May23_05-22-33_oxford-hard"
