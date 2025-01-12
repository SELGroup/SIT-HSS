# SIT-HSS
This repository contains the official code for the paper "Hierarchical Superpixel Segmentation via Structural Information Theory".

### Installation
To install the required libraries, use the following command:
```bash
pip install -r requirements.txt
```
## Usage

### Running the Script
To execute the segmentation pipeline:
```bash
python main_SE.py --t <threshold> --SE_t <se_threshold> --multi_scale <true_or_false> --target_size <sizes>
```

### Parameters
- `--t`: Threshold value for segmentation (default: `0.1`).
- `--SE_t`: SE threshold (default: `2e-7`).
- `--multi_scale`: Whether to use multi-scale segmentation (`True` or `False`, default: `True`).
- `--target_size`: Target size(s) for multi-scale segmentation (default: `[100,200,300,400,500,600,800,1000,1200,1500]`).

### Example Command
```bash
python main_SE.py --t 0.15 --SE_t 3e-7 --multi_scale True --target_size [100,200,300]
```

### Output
Processed images are saved under `./imgs/`, organized by the original image name and target sizes.
