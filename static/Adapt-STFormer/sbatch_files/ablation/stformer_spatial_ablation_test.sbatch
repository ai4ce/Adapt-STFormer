#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=01:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=torch 
#SBATCH --constraint="rtx8000"
#SBATCH --account=pr_130_tandon_advanced

module purge

singularity exec --nv \
    --overlay /scratch/yl9727/my_env/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c \
    "source /ext3/env.sh; \
		python svpr_main.py \
		--arch stformer \
		--mode test \
		--split test \
		--trunc_te 8 --freeze_te 1 \
		--nNeg 5 --seqL 1 \
		--cacheBatchSize 1 \
		--optim ADAM \
		--descType raw_384\
		--img_shape 384 384 \
		--outDims 24576 \
		--backbone cct384 \
		--dataset nuscene \
		--part only_spatial \
		--spatial_ablation \
		--resume data/runs/stformer_cct384_Aug13_00-54-30_oxford-easy"