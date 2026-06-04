# Flexible and Efficient Spatio-Temporal Transformer for Sequential Visual Place Recognition

Yu Kiu (Idan) Lau, Chao Chen, Ge Jin, Chen Feng

## Training the Proposed Model

To train the Adapt-STFormer model, use the SLURM batch script located at:
```
Adapt-STFormer/sbatch_files/train/cct_adapt_stformer_train.sbatch
```

## Dataset Generation

This repository includes dataset generation scripts for **nuScenes only**. The remaining datasets (Oxford RobotCar, Nordland) can be found in other repositories as they were previously implemented elsewhere.

## Understanding structFiles

The `Adapt-STFormer/structFiles` directory contains preprocessed database files essential for training and evaluation:

### How to Read structFiles

The structFiles format is based on the seqNet implementation [[source](https://github.com/oravus/seqNet/issues/3#issuecomment-1006598225)]. The `.db` files contain preprocessed dataset information including:

- **Image file names**: Paths to individual images in the sequence
- **Image locations**: File system paths or dataset indices for each image

and more...

### Usage
The structFiles are automatically loaded during training using the `--dataset` parameter. The system reads the appropriate database and sequence bounds files based on the specified dataset configuration.

## Implementation Disclaimer

**Important Note on Deformable Transformer Encoder (DTE) Usage:**

The implementation in `Adapt-STFormer/svpr/models/modified_multi_scale_deformable_attn_function.py` uses the Deformable Transformer Encoder (DTE) mechanism in a way that deviates from the original intended usage. While this approach may be considered "incorrect" according to the original design specifications, our experiments showed that it achieved better performance results for our specific use case. The modification involves manipulating the sampling locations and attention weights in a non-standard way. 

## Citation
If you use this work, please cite our paper:

```bibtex
@inproceedings{lau2025flexible,
  title     = {Flexible and Efficient Spatio-Temporal Transformer for Sequential Visual Place Recognition},
  author    = {Lau, Yu Kiu and Chen, Chao and Jin, Ge and Feng, Chen},
  booktitle = {2026 IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2025}
}
```