#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
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
		--arch adapt_stformer \
		--pooling seqvlad \
		--mode test \
		--split test \
		--trunc_te 8 --freeze_te 1 \
		--seqL 5 \
		--cacheBatchSize 1 \
		--ckpt best \
		--descType raw_384 \
		--img_shape 384 384 \
		--outDims 24576 \
		--backbone cct384 \
		--dataset nuscene \
		--spatial_ablation \
		--resume data/runs/adapt_stformer_cct384_Aug11_23-34-14_oxford-easy"

# singularity exec --nv \
#     --overlay /scratch/yl9727/my_env/overlay-15GB-500K.ext3:ro \
#     /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
#     /bin/bash -c \
#     "source /ext3/env.sh; \
#     python svpr_main.py \
# 		--arch adapt_stformer \
# 		--pooling seqvlad \
# 		--mode test \
# 		--split test \
# 		--trunc_te 8 --freeze_te 1 \
# 		--nNeg 5 --seqL 3 \
# 		--cacheBatchSize 1 \
# 		--optim ADAM \
# 		--lr 0.00001 \
# 		--descType raw_384 \
# 		--img_shape 384 384 \
# 		--outDims 24576 \
# 		--backbone cct384 \
# 		--dataset nuscene \
# 		--k_points 8 \
# 		--no_recurrent \
# 		--no_seqgem \
# 		--resume data/runs/adapt_stformer_cct384_Aug09_14-38-41_oxford-easy_norecurrent"


# singularity exec --nv \
#     --overlay /scratch/yl9727/my_env/overlay-15GB-500K.ext3:ro \
#     /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
#     /bin/bash -c \
#     "source /ext3/env.sh; \
#     python svpr_main.py \
# 		--arch adapt_stformer \
# 		--pooling seqvlad \
# 		--mode test \
# 		--split test \
# 		--trunc_te 8 --freeze_te 1 \
# 		--nNeg 5 --seqL 1 \
# 		--cacheBatchSize 1 \
# 		--optim ADAM \
# 		--lr 0.00001 \
# 		--descType raw_384 \
# 		--img_shape 384 384 \
# 		--outDims 24576 \
# 		--backbone cct384 \
# 		--dataset nuscene \
# 		--k_points 8 \
# 		--no_recurrent \
# 		--no_seqgem \
# 		--resume data/runs/adapt_stformer_cct384_Aug09_14-38-41_oxford-easy_norecurrent"


