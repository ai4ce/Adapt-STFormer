#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=01:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=torch 
#SBATCH --constraint="rtx8000"
#SBATCH --account=pr_110_tandon_priority

module purge

singularity exec --nv \
    --overlay /scratch/yl9727/my_env/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c \
    "source /ext3/env.sh; \
		python svpr_main.py \
		--arch casenet \
		--mode test \
		--split test \
		--seqL 5 \
		--cacheBatchSize 1 \
		--descType raw_322\
		--img_shape 322 322 \
		--outDims 10752 \
		--dataset oxford-hard \
		--ckpt external_pretrain \
		--resume casenet_pretrained/casevpr_322.pth"

singularity exec --nv \
    --overlay /scratch/yl9727/my_env/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c \
    "source /ext3/env.sh; \
		python svpr_main.py \
		--arch casenet \
		--mode test \
		--split test \
		--seqL 5 \
		--cacheBatchSize 1 \
		--descType raw_322\
		--img_shape 322 322 \
		--outDims 10752 \
        --pcaDim 4096 \
		--dataset nordland-sw \
		--ckpt external_pretrain \
		--resume casenet_pretrained/casevpr_224.pth"


singularity exec --nv \
    --overlay /scratch/yl9727/my_env/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c \
    "source /ext3/env.sh; \
        python svpr_main.py \
        --arch casenet \
        --mode test \
        --split test \
        --seqL 5 \
        --cacheBatchSize 1 \
        --descType raw_322\
        --img_shape 322 322 \
        --outDims 10752 \
        --dataset nordland-sw \
        --ckpt external_pretrain \
        --pca_dim 4096 \
		--vpr_application \
        --resume casenet_pretrained/casevpr_322.pth"