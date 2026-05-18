#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=03:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=torch 
#SBATCH --constraint="rtx8000"
#SBATCH --account=pr_110_tandon_priority

module purge

#NOTE: Oxford-easy!

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
		--dataset oxford-easy \
		--ckpt external_pretrain \
		--vpr_application \
      	--pca_dim 4096 \
		--resume /scratch/yl9727/adaptiveNet/AdaptiveNet/checkpoints/ours/oxford2.pth"


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
		--dataset oxford-easy \
		--ckpt external_pretrain \
		--vpr_application \
		--pca_dim 4096 \
		--resume casenet_pretrained/casevpr_322.pth"


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
		--trunc_te 8 --freeze_te 1 \
		--seqL 5 \
		--cacheBatchSize 1 \
		--descType raw_384 \
		--img_shape 384 384 \
		--outDims 24576 \
		--backbone cct384 \
		--dataset oxford-easy \
		--ckpt best \
		--vpr_application \
		--pca_dim 4096 \
		--resume /scratch/yl9727/adaptiveNet/AdaptiveNet/data/runs/adapt_stformer_cct384_May14_20-09-59_oxford-easy"


#NOTE: Oxford-hard!

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
		--dataset oxford-hard \
		--ckpt best \
		--vpr_application \
		--pca_dim 4096 \
		--resume data/runs/stformer_cct384_Sep06_23-58-01_oxford-hard"

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
		--vpr_application \
		--pca_dim 4096 \
		--resume casenet_pretrained/casevpr_322.pth"

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
		--trunc_te 8 --freeze_te 1 \
		--nNeg 5 --seqL 5 \
		--cacheBatchSize 1 \
		--descType raw_384 \
		--img_shape 384 384 \
		--outDims 24576 \
		--backbone cct384 \
		--dataset oxford-hard \
		--ckpt best \
		--vpr_application \
		--pca_dim 4096 \
		--resume /scratch/yl9727/adaptiveNet/AdaptiveNet/data/runs/adapt_stformer_cct384_May23_11-47-56_oxford-hard"



#NOTE: Nuscene!

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
		--dataset nuscene \
		--ckpt external_pretrain \
		--vpr_application \
      	--pca_dim 4096 \
		--resume /scratch/yl9727/adaptiveNet/AdaptiveNet/checkpoints/ours/oxford2.pth"


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
		--dataset nuscene \
		--ckpt external_pretrain \
		--vpr_application \
		--pca_dim 4096 \
		--resume casenet_pretrained/casevpr_322.pth"


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
		--trunc_te 8 --freeze_te 1 \
		--seqL 5 \
		--cacheBatchSize 1 \
		--descType raw_384 \
		--img_shape 384 384 \
		--outDims 24576 \
		--backbone cct384 \
		--dataset nuscene \
		--ckpt best \
		--vpr_application \
		--pca_dim 4096 \
		--resume /scratch/yl9727/adaptiveNet/AdaptiveNet/data/runs/adapt_stformer_cct384_May14_20-09-59_oxford-easy"
