# UNET_BTS

INITIAL DOCX
- Project title and description

    Lightweight U-Net for Brain Tumor Segmentation (BraTS 2020)

    This project implements a U-Net model optimized for low-resource GPUs (NVIDIA GTX 1660 Ti, 6GB VRAM) to perform brain tumor segmentation on the BraTS 2020 dataset.
    We compare our results against the nnU-Net framework (Fabian Isensee et al., BraTS 2020 winner), highlighting trade-offs between resource usage and segmentation performance.

2. Project Structure

UNET_BTS
│
├── demo.ipynb              # Main notebook: data prep, training, evaluation, visualization
├── README.md               # Project overview, instructions, results
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignore large files, checkpoints, OS junk
├── .gitattributes          # Git LFS, notebook rendering, binary files
│
├── data/                   # Dataset (BraTS 2020)
│   ├── raw/                # Original MRI scans (ignored by git)
│   └── processed/          # Preprocessed 2D slices (ignored by git)
│
├── models/                 # Trained checkpoints (ignored by git)
│   ├── unet_baseline.pth
│   └── unet_optimized.pth
│
├── results/                # Metrics and visualizations (ignored by git)
│   ├── metrics.csv
│   └── sample_predictions/
│
└── utils/                  # Optional helper scripts
    ├── unet.py             # U-Net model definition
    ├── data_loader.py      # Dataset + preprocessing functions
    └── metrics.py          # Dice, HD95 <-- to be checked


3. Setup

    a. Clone repo:
        - git clone https://github.com/Blaq2011/UNET_BTS.git

    b. Install requirements
        - pip install -r requirements.txt

    c. Get Dataset
        - Download the BraTS 2020 dataset from: https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation?resource=download-directory 
        - Extract dataset in data/raw/.
        - Preprocessed 2D slices are saved automatically in data/processed/


4. Usage
    - Open the main notebook (demo.ipynb) and run all cells

The notebook includes:
<!-- Data preprocessing (2D slices, normalization, augmentation)
Training baseline U-Net
Training optimized U-Net
Evaluation (Dice, IoU, Hausdorff distance)
Visualization of predictions vs. ground truth -->

5. Results

<!-- | Model             | Dice (Whole Tumor) | IoU  | Notes                        |
| ----------------- | ------------------ | ---- | ---------------------------- |
| Baseline U-Net    | XX.XX              | XX.X | Small filters, limited GPU   |
| Optimized U-Net   | XX.XX              | XX.X | With augmentations + dropout |
| nnU-Net (Fabian+) | \~0.88–0.90        | --   | BraTS 2020 winner            | -->


<!-- Qualitative Results
    (Example figure to be added here)

    Input MRI | Ground Truth | Baseline Prediction | Optimized Prediction -->

6. References

Fabian Isensee et al. (2020), nnU-Net for Brain Tumor Segmentation. https://arxiv.org/abs/2011.00848 
BraTS 2020 Challenge: https://www.med.upenn.edu/cbica/brats2020/


Note: This implementation is designed for educational purposes under GPU memory constraints. Performance is not expected to match state-of-the-art methods.
