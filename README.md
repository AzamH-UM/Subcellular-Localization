# Subcellular-Localization
Experiments on deep learning based subcellular localization of proteins using structural information

## Folders
- data
  - Data.csv
  - deeploc_af2_df.xlsx
    - Excel sheet with all sequences and labels for training and testing
  - esm_feats.npy
- utils

## Scripts
- run_esm.ipynb
  - train ESM2 on sequences of length 500 
- evaluate_esm.ipynb
  - Evalutae ESM2 and other trained models
- train_cnn.ipynb
  - Train 3D AlexNet and VGG16 on voxellized proteins
- run_fusion.ipynb
  - Train mid level fusion model with ESM2 and 3D AlexNet/VGG16
- voxel_models.py
  - Contains classes for 3D models
- midfusion_mlp_cnn.py
- mlp_esm..ipynb
- pure_cnn.py
