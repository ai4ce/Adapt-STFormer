#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=15:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=torch
#SBATCH --account=pr_130_tandon_advanced

module purge

singularity exec --nv \
    --overlay /scratch/yl9727/my_env/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
        source /ext3/env.sh;
        python svpr_main.py \
        --mode train \
        --arch seqnet \
        --dataset nordland-sw \
        --seqL 10 \
        --w 5 \
        --seed 43 \
        --outDims 4096 \
        --expName 'w5' \
        --descType netvlad-pytorch \
        --batchSize 16 \
        --cacheBatchSize 16 \
        --nEpochs 200
        --nNeg 10"
