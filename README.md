# UNET_BTS

INITIAL DOCX
- Project title and description

Title: Resource-Conscious 3D U-Net Models for Brain Tumor Segmentation: An Ablation Study

Abstract:
    Accurate brain tumor segmentation is critical for diagnosis and treatment planning, but current state-of-the-art
methods like nnU-Net are computationally demanding. This
work explores lightweight 3D U-Net variants tailored for
resource-limited settings. Using the BraTS 2020 dataset,
we compared three preprocessing pipelines and conducted
ablations on baseline and optimized models, incorporat-
ing residual blocks, attention gates, deep supervision, nor-
malization refinements, dropout tuning, and class-weighted
loss. The best model achieved Dice scores of 0.880 (WT),
0.812 (TC), and 0.691 (ET), showing consistent improve-
ments over the baseline while still remaining computation-
ally efficient. These results highlight the trade-off between
efficiency and absolute accuracy, and point to practical
strategies for segmentation under limited GPU resources.

2. Project Structure

UNET_BTS/
├── .gitattributes
├── .gitignore
├── demo.ipynb
├── README.md
├── requirements.txt

|   
└── .ipynb_checkpoints/
├── demo-checkpoint.ipynb
├── README-checkpoint.md
├── requirements-checkpoint.txt
├── test-checkpoint.ipynb
└── UNET Test-checkpoint.ipynb
|       
+---data                                                                    <--#Git Ignored
|   +---processed
|   |   \---cache
|   \---raw
+---models
|   +---.ipynb_checkpoints
|   +---model comparison
|   |       Model2_noDrop_P2_s5.pth
|   |       optModel1_classweight_P2_s5.pth
|   |       
|   \---pipeline abalation
|           unet_P1_s0.pth
|           unet_P1_s1.pth
|           unet_P1_s2.pth
|           unet_P2_s0.pth
|           unet_P2_s1.pth
|           unet_P2_s2.pth
|           unet_P3_s0.pth
|           unet_P3_s1.pth
|           unet_P3_s2.pth
|           
+---results
|   +---.ipynb_checkpoints
|   |       all_pipelines_history-checkpoint.csv
|   |       all_results-checkpoint.csv
|   |       P1_history_0-checkpoint.csv
|   |       
|   +---Images
|   |   \---report images
|   |           Allpipelines_TrainVal_loss.png
|   |           ModelComparison1.png
|   |           pipelines visuals.png
|   |           
|   +---model comparison
|   |   +---base
|   |   |       all_pipelines_history.csv
|   |   |       all_results.csv
|   |   |       P2_history_5.csv
|   |   |       
|   |   \---optimized
|   |       +---model1
|   |       |   |   Model_1_history.csv
|   |       |   |   Model_1_P2_history_5.csv
|   |       |   |   Model_1_results.csv
|   |       |   |   
|   |       |   \---.ipynb_checkpoints
|   |       |           Model_1_history-checkpoint.csv
|   |       |           Model_1_P2_history_5-checkpoint.csv
|   |       |           Model_1_results-checkpoint.csv
|   |       |           
|   |       +---model2
|   |       |       Model_2_all_results.csv
|   |       |       Model_2_history.csv
|   |       |       Model_2_P2_history_5.csv
|   |       |       
|   |       +---model2_Nodropout
|   |       |       Model_2_Nodropout_all_results.csv
|   |       |       Model_2_Nodropout_history.csv
|   |       |       Model_2_Nodropout_P2_history_5.csv
|   |       |       
|   |       \---model2_Nodropout_classWeights
|   |               Model2_classweight_history.csv
|   |               Model2_classweight_results.csv
|   |               P2_history_5.csv
|   |               
|   +---pipeline ablation
|   |       all_pipelines_history.csv
|   |       all_results.csv
|   |       P1_history_0.csv
|   |       P1_history_1.csv
|   |       P1_history_2.csv
|   |       P2_history_0.csv
|   |       P2_history_1.csv
|   |       P2_history_2.csv
|   |       P3_history_0.csv
|   |       P3_history_1.csv
|   |       P3_history_2.csv
|   |       
|   \---sample_predictions
\---utils
    |   data_loader.py
    |   metrics.py
    |   run_train_eval.py
    |   seeding.py
    |   train_unet.py
    |   unet.py
    |   visualize.py
    |   
    +---.ipynb_checkpoints
    |       data_loader-checkpoint.py
    |       evaluate_unet-checkpoint.py
    |       metrics-checkpoint.py
    |       run_train_eval-checkpoint.py
    |       seeding-checkpoint.py
    |       train_unet-checkpoint.py
    |       unet-checkpoint.py
    |       visualize-checkpoint.py
    |       
    \---__pycache__
            ablation_loader.cpython-312.pyc
            data_loader.cpython-312.pyc
            evaluate_unet.cpython-312.pyc
            metrics.cpython-312.pyc
            run_train_eval.cpython-312.pyc
            seeding.cpython-312.pyc
            train_unet.cpython-312.pyc
            unet.cpython-312.pyc
            unet_test.cpython-312.pyc
            visualize.cpython-312.pyc
            



3. Setup

    a. Clone repo:
        - git clone https://github.com/Blaq2011/UNET_BTS.git

    b. Install requirements
        - pip install -r requirements.txt

    c. Create directories and folders ignored by git (see project structure).
   
    d. Get Dataset
        - Download the BraTS 2020 dataset from: https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation?resource=download-directory 
        - Extract dataset in data/raw/.
        - Preprocessed 2D slices are saved automatically in data/processed/cache


4. Usage
    - Open the main notebook (demo.ipynb) and run all cells <-- TO BE CHECKED







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

[1] Mohammad Havaei, Nicolas Guizard, Nicolas Chapados,
and Yoshua Bengio. Brain tumor segmentation with deep
neural networks. Medical Image Analysis, 35:18–31, 2017.
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
Deep residual learning for image recognition. In Proceed-
ings of the IEEE Conference on Computer Vision and Pattern
Recognition, pages 770–778, 2016.
[3] Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation
networks. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), pages
7132–7141, 2018.
[4] Fabian Isensee, Paul F. Jaeger, Peter M. Full, Philipp Voll-
muth, and Klaus H. Maier-Hein. nnu-net for brain tumor
segmentation, 2020.
[5] Fabian Isensee, Paul F. Jaeger, Simon A. A. Kohl, Jens Pe-
tersen, and Klaus H. Maier-Hein. nnu-net: a self-configuring
method for deep learning-based biomedical image segmen-
tation. Nature Methods, 18(2):203–211, 2021.
[6] Chen-Yu Lee, Saining Xie, Patrick Gallagher, Zhengyou
Zhang, and Zhuowen Tu. Deeply-supervised nets. In Pro-
ceedings of the 18th International Conference on Artificial
Intelligence and Statistics, pages 562–570, 2015.
[7] Bastian H. Menze, ´Andras Jakab, Stefan Bauer, Jayashree
Kalpathy-Cramer, Keyvan Farahani, John Kirby, Yvette Bur-
ren, Nicolas Porz, Jens Slotboom, Roland Wiest, and Koen
van Leemput. The multimodal brain tumor image segmen-
tation benchmark (brats). IEEE Transactions on Medical
Imaging, 34(10):1993–2024, 2015.
[8] Ozan Oktay, Jo Schlemper, Lo¨ıc Le Folgoc, Matthew
Lee, Mattias P. Heinrich, Kazunari Misawa, Kensaku Mori,
Steven McDonagh, Nils Y. Hammerla, Bernhard Kainz, Ben
Glocker, and Daniel Rueckert. Attention u-net: Learn-
ing where to look for the pancreas. arXiv preprint
arXiv:1804.03999, 2018.
[9] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net:
Convolutional networks for biomedical image segmentation.
arXiv preprint arXiv:1505.04597, 2015.
[10] Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya
Sutskever, and Ruslan Salakhutdinov. Dropout: A simple
way to prevent neural networks from overfitting. Journal of
Machine Learning Research, 15(1):1929–1958, 2014.
[11] Yuxin Wu and Kaiming He. Group normalization. In Pro-
ceedings of the European Conference on Computer Vision
(ECCV), volume 11217 of Lecture Notes in Computer Sci-
ence, pages 3–19. Springer, Cham, 2018.


Note: This implementation is designed for educational purposes under GPU memory constraints. Performance is not expected to match state-of-the-art methods.
