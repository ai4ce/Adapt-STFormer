#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=01:00:00
#SBATCH --mem=350GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=torch 
#SBATCH --constraint="a100"
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
		--arch jist \
		--nNeg 5 --seqL 5 \
		--cacheBatchSize 16 \
		--descType raw_384 \
		--img_shape 384 384 \
		--outDims 512 \
		--backbone ResNet18 \
		--dataset nordland-sf \
		--ckpt external_pretrain \
		--resume checkpoints/JIST/JIST_r18_512_seqgem.pth"
