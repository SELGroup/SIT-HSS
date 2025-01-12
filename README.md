# SIT-HSS

## Project Overview

This project implements a segmentation pipeline using Python. It processes images from specified datasets, applies segmentation algorithms, and saves the processed images. The project supports multi-scale segmentation with configurable parameters.

## Features
- Supports multiple datasets (`BSDS`, `SBD`, `PASCAL-S`).
- Multi-scale segmentation with configurable target sizes.
- Boundary visualization for segmented images.
- Outputs segmented images into organized directories.

## Requirements

### Libraries
This project requires the following Python libraries:
- `PIL` (from `Pillow`)
- `cv2` (from `OpenCV`)
- `pandas`
- `matplotlib`
- `torch`
- `torch_scatter`
- `numpy`
- `scipy`
- `skimage`

### Installation
To install the required libraries, use the following command:
```bash
pip install -r requirements.txt
```

Create a `requirements.txt` file if it doesn’t exist:
```plaintext
Pillow
opencv-python
pandas
matplotlib
torch
torch-scatter
numpy
scipy
scikit-image
```

## Usage

### Dataset Structure
Ensure your datasets follow the structure below:

```
BSDS500/
  images/
  groundTruth/
SBD/
  images/
  groundTruth/
PASCAL-S/
  images/
  groundTruth/
```

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

## Code Structure
- **`get_args()`**: Parses command-line arguments.
- **`draw_img(img, seg)`**: Generates boundary visualizations for segmented images.
- **`start(data, gt, cfg)`**: Processes a dataset and saves segmented images.
- **`main()`**: Entry point to run the segmentation pipeline.

## Contribution
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a Pull Request.

## License
This project is licensed under the MIT License. See `LICENSE` for details.
