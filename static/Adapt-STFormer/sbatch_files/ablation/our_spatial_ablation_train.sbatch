#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=torch 
#SBATCH --constraint="a100|rtx8000"
#SBATCH --account=pr_130_tandon_advanced

module purge

singularity exec --nv \
    --overlay /scratch/yl9727/my_env/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
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
		--spatial_ablation \
		--weightDecay 0 \
		--resume data/runs/adapt_stformer_cct384_Aug11_23-34-14_oxford-easy"
