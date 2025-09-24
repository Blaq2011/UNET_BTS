import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", context="talk")

import nilearn as nl
import nilearn.plotting as nlplt

# =================Viewing sample image data ==================

def visualize_sample(Test_filepath,test_t1_image,test_t1ce_image,test_t2_image,test_flair_image, test_mask, volume_show):
    fig , (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize = (20, 10))

    fig.suptitle('Sample data (Axial Slice) showing different varieties of Scans and the Mask'
                , y=0.70, fontsize=20 )
    
    ax1.imshow(test_t1_image[:,:,volume_show], cmap = "gray")
    ax1.set_title("t1 image")
    ax1.grid(False)
    
    ax2.imshow(test_t1ce_image[:,:,volume_show], cmap = "gray")
    ax2.set_title("t1ce image")
    ax2.grid(False)
    
    ax3.imshow(test_t2_image[:,:,volume_show], cmap = "gray")
    ax3.set_title("t2 image")
    ax3.grid(False)
    
    ax4.imshow(test_flair_image[:,:,volume_show], cmap = "gray")
    ax4.set_title("flair image")
    ax4.grid(False)
    
    ax5.imshow(test_mask[:,:,volume_show])
    ax5.set_title("seg mask")
    ax5.grid(False)
  
    nl_test_t1_image = nl.image.load_img(Test_filepath + "BraTS20_Training_001_t1.nii")
    nl_test_mask_image = nl.image.load_img(Test_filepath + "BraTS20_Training_001_seg.nii")
    
    # nlplt.plot_epi(nl_test_t1_image, title="T1 Tumor Image in Different orientaions")
    nlplt.plot_roi(nl_test_mask_image,bg_img=nl_test_t1_image , title="Highlighted ROI (Tumor) in different orientations")
 


# ===================================================================================
#Visualizing the different pipelines 
def visualize_patient_consistency(P1, P2, P3, patient_idx=0, slice_axis=0, seed=1234):
    """
    Visualize dataset outputs for a given patient index.
    Each row = pipeline (P1, P2, P3).
    Each column = modality (FLAIR, T1, T1CE, T2, Mask).
    Shows exactly what the model sees (post-preprocessing).
    
    slice_axis: 0=axial(z), 1=coronal(y), 2=sagittal(x)
    """

    datasets = [P1, P2, P3]
    row_labels = ["P1 (on-the-fly)", "P2 (cached vol)", "P3 (cached patches)"]
    col_titles = ["FLAIR", "T1", "T1CE", "T2", "Mask"]

    rows = []
    for ds in datasets:
        # get the preprocessed patch & its mask

        if seed is not None:
            random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)# reset RNG so both datasets sample the same patch coords
            img_patch, mask_patch = ds[patient_idx]
        else:
            img_patch, mask_patch = ds[patient_idx]
            
        img_patch, mask_patch = img_patch.numpy(), mask_patch.numpy()
        mask_patch = np.argmax(mask_patch, axis=0)

        # choose middle slice
        if slice_axis == 0:
            mid = img_patch.shape[1] // 2
            imgs = [img_patch[i, mid, :, :] for i in range(4)]
            m = mask_patch[mid, :, :]
        elif slice_axis == 1:
            mid = img_patch.shape[2] // 2
            imgs = [img_patch[i, :, mid, :] for i in range(4)]
            m = mask_patch[:, mid, :]
        elif slice_axis == 2:
            mid = img_patch.shape[3] // 2
            imgs = [img_patch[i, :, :, mid] for i in range(4)]
            m = mask_patch[:, :, mid]
        else:
            raise ValueError("slice_axis must be 0 (axial), 1 (coronal), or 2 (sagittal)")

        rows.append(imgs + [m])

    # plotting
    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    if seed is not None:
        fig.suptitle(f"Patient_deterministic {patient_idx}, axis={slice_axis}", fontsize=16)
    else:
        fig.suptitle(f"Patient_random {patient_idx}, axis={slice_axis}", fontsize=16)

    for r in range(3):
        for c in range(5):
            cmap = "gray" if c < 4 else "nipy_spectral"
            axes[r, c].imshow(rows[r][c], cmap=cmap)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(col_titles[c], fontsize=12)

    plt.tight_layout(rect=(0.08, 0.03, 1.0, 0.92))

    for r, label in enumerate(row_labels):
        pos = axes[r, 0].get_position()
        y_center = pos.y0 + pos.height / 2
        fig.text(
            0.02, y_center, label, va="center", ha="left",
            rotation=90, fontsize=12, fontweight="bold"
        )
        
    if seed is not None:
        out_path = f"results/images/Patient_deterministic-{patient_idx}_consistency.png"
    else:
        out_path = f"results/images/Patient_random-{patient_idx}_consistency.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.show()



# ===================================================================================

def plot_losses_per_seed(csv_file: str):
    """
    Plot training vs validation loss for each pipeline, grouped by seed.
    Each seed will produce its own figure.
    """
    df = pd.read_csv(csv_file)
    seeds = df["seed"].unique()

    for seed in seeds:
        sub = df[df["seed"] == seed]

        plt.figure(figsize=(10, 6))

        # Training loss (solid)
        sns.lineplot(
            data=sub,
            x="epoch", y="train_loss",
            hue="pipeline", style="pipeline",
            markers=True, dashes=False,
            legend="full", alpha=0.8
        )

        # Validation loss (dashed)
        sns.lineplot(
            data=sub,
            x="epoch", y="val_loss",
            hue="pipeline", style="pipeline",
            markers=False, dashes=True,
            legend=False, alpha=0.8
        )

        plt.title(f"Seed {seed} - Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(title="Pipeline")
        plt.tight_layout()
        plt.show()


def plot_loss_summary(csv_file: str):
    """
    Plot mean ± std of training and validation loss across seeds for each pipeline.
    Produces two figures: one for training loss, one for validation loss.
    """
    df = pd.read_csv(csv_file)

    # Aggregate mean/std across seeds
    summary = (
        df.groupby(["pipeline", "epoch"])
        .agg(
            train_mean=("train_loss", "mean"),
            train_std=("train_loss", "std"),
            val_mean=("val_loss", "mean"),
            val_std=("val_loss", "std")
        )
        .reset_index()
    )

    # --- Training Loss ---
    plt.figure(figsize=(10, 6))
    for pipeline in summary["pipeline"].unique():
        sub = summary[summary["pipeline"] == pipeline]
        plt.plot(sub["epoch"], sub["train_mean"], label=pipeline, linewidth=2)
        plt.fill_between(
            sub["epoch"],
            sub["train_mean"] - sub["train_std"],
            sub["train_mean"] + sub["train_std"],
            alpha=0.2
        )
    plt.title("Training Loss Across Pipelines (Averaged Over Seeds)")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.legend(title="Pipeline")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # --- Validation Loss ---
    plt.figure(figsize=(10, 6))
    for pipeline in summary["pipeline"].unique():
        sub = summary[summary["pipeline"] == pipeline]
        plt.plot(sub["epoch"], sub["val_mean"], label=pipeline, linewidth=2)
        plt.fill_between(
            sub["epoch"],
            sub["val_mean"] - sub["val_std"],
            sub["val_mean"] + sub["val_std"],
            alpha=0.2
        )
    plt.title("Validation Loss Across Pipelines (Averaged Over Seeds)")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend(title="Pipeline")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
