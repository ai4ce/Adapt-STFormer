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
		--arch stformer \
		--mode test \
		--split test \
		--trunc_te 4 --freeze_te -1 \
		--trunc_te_tatt 4 --freeze_te_tatt -1 \
		--seqL 5 \
		--cacheBatchSize 1 \
		--descType raw_384\
		--img_shape 384 384 \
		--outDims 24576 \
		--backbone cct384 \
		--rel_pos_temporal \
		--abs_pos_embed \
		--pca_dim 4096 \
		--dataset nuscene \
		--ckpt best \
		--vpr_application \
		--resume /scratch/yl9727/trained_models_james/stformer/oxford2.pth"


singularity exec --nv \
    --overlay /scratch/yl9727/my_env/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c \
    "source /ext3/env.sh; \
		python svpr_main.py \
		--arch stformer \
		--mode test \
		--split test \
		--trunc_te 4 --freeze_te -1 \
		--trunc_te_tatt 4 --freeze_te_tatt -1 \
		--seqL 5 \
		--cacheBatchSize 1 \
		--descType raw_384\
		--img_shape 384 384 \
		--outDims 24576 \
		--backbone cct384 \
		--rel_pos_temporal \
		--abs_pos_embed \
		--pca_dim 4096 \
		--dataset oxford-easy \
		--ckpt external_pretrain \
		--vpr_application \
		--resume /scratch/yl9727/trained_models_james/stformer/oxford2.pth"

singularity exec --nv \
    --overlay /scratch/yl9727/my_env/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c \
    "source /ext3/env.sh; \
		python svpr_main.py \
		--arch stformer \
		--mode test \
		--split test \
		--trunc_te 4 --freeze_te -1 \
		--trunc_te_tatt 4 --freeze_te_tatt -1 \
		--seqL 5 \
		--cacheBatchSize 1 \
		--descType raw_384\
		--img_shape 384 384 \
		--outDims 24576 \
		--backbone cct384 \
		--rel_pos_temporal \
		--abs_pos_embed \
		--pca_dim 4096 \
		--dataset oxford-easy \
		--ckpt best \
		--vpr_application \
		--resume /scratch/yl9727/trained_models_james/stformer_cct384_Sep13_11-51-47_oxford-hard"