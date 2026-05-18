#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --mem=200GB
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
		--mode test \
		--split test \
		--arch adapt_stformer \
		--pooling seqvlad \
		--trunc_te 4 --freeze_te 1 \
		--seqL 5 \
		--cacheBatchSize 1 \
		--descType raw_384 \
		--img_shape 384 384 \
		--outDims 24576 \
		--backbone cct384 \
		--dataset nordland-sw \
		--ckpt best \
		--k_points 8 \
        --num_levels 2 \
		--resume data/"

# singularity exec --nv \
#     --overlay /scratch/yl9727/my_env/overlay-15GB-500K.ext3:ro \
#     /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
#     /bin/bash -c \
#     "source /ext3/env.sh; \
# 		python svpr_main.py \
# 		--mode test \
# 		--split test \
# 		--arch adapt_stformer \
# 		--pooling seqvlad \
# 		--trunc_te 8 --freeze_te 1 \
# 		--seqL 3 \
# 		--cacheBatchSize 1 \
# 		--descType raw_384 \
# 		--img_shape 384 384 \
# 		--outDims 24576 \
# 		--backbone cct384 \
# 		--dataset nuscene \
# 		--ckpt best \
# 		--k_points 8 \
#         --num_levels 2 \
# 		--spatial_ablation \
# 		--resume data/runs/adapt_stformer_cct384_Sep15_01-17-23_oxford-nuscene-pairing"

# singularity exec --nv \
#     --overlay /scratch/yl9727/my_env/overlay-15GB-500K.ext3:ro \
#     /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
#     /bin/bash -c \
#     "source /ext3/env.sh; \
# 		python svpr_main.py \
# 		--mode test \
# 		--split test \
# 		--arch adapt_stformer \
# 		--pooling seqvlad \
# 		--trunc_te 8 --freeze_te 1 \
# 		--seqL 1 \
# 		--cacheBatchSize 1 \
# 		--descType raw_384 \
# 		--img_shape 384 384 \
# 		--outDims 24576 \
# 		--backbone cct384 \
# 		--dataset nuscene \
# 		--ckpt best \
# 		--k_points 8 \
#         --num_levels 2 \
# 		--spatial_ablation \
# 		--resume data/runs/adapt_stformer_cct384_Sep15_01-17-23_oxford-nuscene-pairing"

# singularity exec --nv \
#     --overlay /scratch/yl9727/my_env/overlay-15GB-500K.ext3:ro \
#     /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
#     /bin/bash -c \
#     "source /ext3/env.sh; \
# 		python svpr_main.py \
# 		--mode test \
# 		--split test \
# 		--arch adapt_stformer \
# 		--pooling seqvlad \
# 		--trunc_te 8 --freeze_te 1 \
# 		--seqL 5 \
# 		--cacheBatchSize 1 \
# 		--descType raw_384 \
# 		--img_shape 384 384 \
# 		--outDims 24576 \
# 		--backbone cct384 \
# 		--dataset oxford-hard \
# 		--ckpt best \
# 		--k_points 8 \
#         --num_levels 2 \
# 		--spatial_ablation \
# 		--resume data/runs/adapt_stformer_cct384_Sep15_01-03-02_oxford-hard"

# singularity exec --nv \
#     --overlay /scratch/yl9727/my_env/overlay-15GB-500K.ext3:ro \
#     /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
#     /bin/bash -c \
#     "source /ext3/env.sh; \
# 		python svpr_main.py \
# 		--mode test \
# 		--split test \
# 		--arch adapt_stformer \
# 		--pooling seqvlad \
# 		--trunc_te 8 --freeze_te 1 \
# 		--seqL 3 \
# 		--cacheBatchSize 1 \
# 		--descType raw_384 \
# 		--img_shape 384 384 \
# 		--outDims 24576 \
# 		--backbone cct384 \
# 		--dataset oxford-hard \
# 		--ckpt best \
# 		--k_points 8 \
#         --num_levels 2 \
# 		--spatial_ablation \
# 		--resume data/runs/adapt_stformer_cct384_Sep15_01-03-02_oxford-hard"


# singularity exec --nv \
#     --overlay /scratch/yl9727/my_env/overlay-15GB-500K.ext3:ro \
#     /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
#     /bin/bash -c \
#     "source /ext3/env.sh; \
# 		python svpr_main.py \
# 		--mode test \
# 		--split test \
# 		--arch adapt_stformer \
# 		--pooling seqvlad \
# 		--trunc_te 8 --freeze_te 1 \
# 		--seqL 1 \
# 		--cacheBatchSize 1 \
# 		--descType raw_384 \
# 		--img_shape 384 384 \
# 		--outDims 24576 \
# 		--backbone cct384 \
# 		--dataset oxford-hard \
# 		--ckpt best \
# 		--k_points 8 \
#         --num_levels 2 \
# 		--spatial_ablation \
# 		--resume data/runs/adapt_stformer_cct384_Sep15_01-03-02_oxford-hard"

# singularity exec --nv \
#     --overlay /scratch/yl9727/my_env/overlay-15GB-500K.ext3:ro \
#     /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
#     /bin/bash -c \
#     "source /ext3/env.sh; \
# 		python svpr_main.py \
# 		--mode test \
# 		--split test \
# 		--arch adapt_stformer \
# 		--pooling seqvlad \
# 		--trunc_te 8 --freeze_te 1 \
# 		--seqL 5 \
# 		--cacheBatchSize 1 \
# 		--descType raw_384 \
# 		--img_shape 384 384 \
# 		--outDims 24576 \
# 		--backbone cct384 \
# 		--dataset oxford-hard \
# 		--ckpt best \
# 		--pca_dim 512 \
# 		--k_points 8 \
#         --num_levels 2 \
# 		--resume data/runs/adapt_stformer_cct384_Sep11_10-52-08_oxford-hard"


# #NUSCENE!

# singularity exec --nv \
#     --overlay /scratch/yl9727/my_env/overlay-15GB-500K.ext3:ro \
#     /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
#     /bin/bash -c \
#     "source /ext3/env.sh; \
# 		python svpr_main.py \
# 		--mode test \
# 		--split test \
# 		--arch adapt_stformer \
# 		--pooling seqvlad \
# 		--trunc_te 8 --freeze_te 1 \
# 		--seqL 5 \
# 		--cacheBatchSize 1 \
# 		--descType raw_384 \
# 		--img_shape 384 384 \
# 		--outDims 24576 \
# 		--backbone cct384 \
# 		--dataset nuscene \
# 		--ckpt best \
# 		--k_points 8 \
#         --num_levels 2 \
# 		--no_seqgem \
# 		--resume data/runs/adapt_stformer_cct384_Sep11_02-29-35_oxford-easy"

# singularity exec --nv \
#     --overlay /scratch/yl9727/my_env/overlay-15GB-500K.ext3:ro \
#     /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
#     /bin/bash -c \
#     "source /ext3/env.sh; \
# 		python svpr_main.py \
# 		--mode test \
# 		--split test \
# 		--arch adapt_stformer \
# 		--pooling seqvlad \
# 		--trunc_te 8 --freeze_te 1 \
# 		--seqL 5 \
# 		--cacheBatchSize 1 \
# 		--descType raw_384 \
# 		--img_shape 384 384 \
# 		--outDims 24576 \
# 		--backbone cct384 \
# 		--dataset nuscene \
# 		--ckpt best \
# 		--pca_dim 4096 \
# 		--k_points 8 \
#         --num_levels 2 \
# 		--vpr_application \
# 		--resume data/runs/adapt_stformer_cct384_Sep11_02-29-35_oxford-easy"

# singularity exec --nv \
#     --overlay /scratch/yl9727/my_env/overlay-15GB-500K.ext3:ro \
#     /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
#     /bin/bash -c \
#     "source /ext3/env.sh; \
# 		python svpr_main.py \
# 		--mode test \
# 		--split test \
# 		--arch adapt_stformer \
# 		--pooling seqvlad \
# 		--trunc_te 8 --freeze_te 1 \
# 		--seqL 5 \
# 		--cacheBatchSize 1 \
# 		--descType raw_384 \
# 		--img_shape 384 384 \
# 		--outDims 24576 \
# 		--backbone cct384 \
# 		--dataset nuscene \
# 		--ckpt best \
# 		--pca_dim 512 \
# 		--k_points 8 \
#         --num_levels 2 \
# 		--resume data/runs/adapt_stformer_cct384_Sep11_02-29-35_oxford-easy"

# # Oxford-easy

# singularity exec --nv \
#     --overlay /scratch/yl9727/my_env/overlay-15GB-500K.ext3:ro \
#     /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
#     /bin/bash -c \
#     "source /ext3/env.sh; \
# 		python svpr_main.py \
# 		--mode test \
# 		--split test \
# 		--arch adapt_stformer \
# 		--pooling seqvlad \
# 		--trunc_te 8 --freeze_te 1 \
# 		--seqL 5 \
# 		--cacheBatchSize 1 \
# 		--descType raw_384 \
# 		--img_shape 384 384 \
# 		--outDims 24576 \
# 		--backbone cct384 \
# 		--dataset oxford-easy \
# 		--ckpt best \
# 		--k_points 8 \
#         --num_levels 2 \
# 		--resume data/runs/adapt_stformer_cct384_Sep11_02-29-35_oxford-easy"

# singularity exec --nv \
#     --overlay /scratch/yl9727/my_env/overlay-15GB-500K.ext3:ro \
#     /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
#     /bin/bash -c \
#     "source /ext3/env.sh; \
# 		python svpr_main.py \
# 		--mode test \
# 		--split test \
# 		--arch adapt_stformer \
# 		--pooling seqvlad \
# 		--trunc_te 8 --freeze_te 1 \
# 		--seqL 5 \
# 		--cacheBatchSize 1 \
# 		--descType raw_384 \
# 		--img_shape 384 384 \
# 		--outDims 24576 \
# 		--backbone cct384 \
# 		--dataset oxford-easy \
# 		--ckpt best \
# 		--k_points 8 \
#         --num_levels 2 \
# 		--pca_dim 4096 \
# 		--vpr_application \
# 		--resume data/runs/adapt_stformer_cct384_Sep11_02-29-35_oxford-easy"

# singularity exec --nv \
#     --overlay /scratch/yl9727/my_env/overlay-15GB-500K.ext3:ro \
#     /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
#     /bin/bash -c \
#     "source /ext3/env.sh; \
# 		python svpr_main.py \
# 		--mode test \
# 		--split test \
# 		--arch adapt_stformer \
# 		--pooling seqvlad \
# 		--trunc_te 8 --freeze_te 1 \
# 		--seqL 5 \
# 		--cacheBatchSize 1 \
# 		--descType raw_384 \
# 		--img_shape 384 384 \
# 		--outDims 24576 \
# 		--backbone cct384 \
# 		--dataset oxford-easy \
# 		--ckpt best \
# 		--k_points 8 \
#         --num_levels 2 \
# 		--pca_dim 512 \
# 		--resume data/runs/adapt_stformer_cct384_Sep11_02-29-35_oxford-easy"


# singularity exec --nv \
#     --overlay /scratch/yl9727/my_env/overlay-15GB-500K.ext3:ro \
#     /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
#     /bin/bash -c \
#     "source /ext3/env.sh; \
# 		python svpr_main.py \
# 		--mode test \
# 		--split test \
# 		--arch adapt_stformer \
# 		--pooling seqvlad \
# 		--trunc_te 8 --freeze_te 1 \
# 		--seqL 5 \
# 		--cacheBatchSize 1 \
# 		--descType raw_384 \
# 		--img_shape 384 384 \
# 		--outDims 24576 \
# 		--backbone cct384 \
# 		--dataset nuscene \
# 		--ckpt latest \
# 		--k_points 8 \
#         --num_levels 2 \
# 		--resume data/runs/adapt_stformer_cct384_Sep11_02-29-35_oxford-easy"


# singularity exec --nv \
#     --overlay /scratch/yl9727/my_env/overlay-15GB-500K.ext3:ro \
#     /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
#     /bin/bash -c \
#     "source /ext3/env.sh; \
# 		python svpr_main.py \
# 		--mode test \
# 		--split test \
# 		--arch adapt_stformer \
# 		--pooling seqvlad \
# 		--trunc_te 8 --freeze_te 1 \
# 		--seqL 5 \
# 		--cacheBatchSize 1 \
# 		--descType raw_384 \
# 		--img_shape 384 384 \
# 		--outDims 24576 \
# 		--backbone cct384 \
# 		--dataset nuscene \
# 		--ckpt latest \
# 		--k_points 8 \
#         --num_levels 2 \
# 		--resume data/runs/adapt_stformer_cct384_Sep12_11-42-55_oxford-easy"
