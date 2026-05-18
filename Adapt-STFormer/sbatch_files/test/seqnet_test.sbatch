#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=torch
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
		--arch seqnet \
		--nNeg 5 --seqL 5 \
		--cacheBatchSize 1 \
		--descType netvlad-pytorch \
		--outDims 4096 \
		--dataset nordland-sw \
		--ckpt best \
		--weightDecay 0 \
		--resume data/runs/seqnet_cct384_Jun01_11-00-15_nordland-sw"
