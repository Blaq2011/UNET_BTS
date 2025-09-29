import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", context="talk")
import torch
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm

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

def plot_losses_per_seed(csv_file: str, colors=("orange", "green", "purple")):
    """
    Plot training vs validation loss for each pipeline, grouped by seed.
    Each seed will produce its own figure.
    """
    df = pd.read_csv(csv_file)
    seeds = df["seed"].unique()

    pipelines = sorted(df["pipeline"].unique())
    palette = {p: c for p, c in zip(pipelines, colors)}
    
    for seed in seeds:
        sub = df[df["seed"] == seed]

        plt.figure(figsize=(10, 6))

        # Training loss (solid)
        sns.lineplot(
            data=sub,
            x="epoch", y="train_loss",
            hue="pipeline", palette=palette, 
            linestyle="-",marker="o",
            legend=False, alpha=0.8
        )

        # Validation loss (dashed)
        sns.lineplot(
            data=sub,
            x="epoch", y="val_loss",
            hue="pipeline", palette=palette,
            linestyle="--",markers=False, 
            legend=False, alpha=0.8
        )

        from matplotlib.lines import Line2D
        legend_lines = [
            Line2D([0], [0], color="black", linestyle="-", label="Training"),
            Line2D([0], [0], color="black", linestyle="--", label="Validation")
        ]
        for p in pipelines:
            legend_lines.append(Line2D([0], [0], color=palette[p], marker="o", label=p))

       
        plt.title(f"Seed {seed} - Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(handles=legend_lines, title="Line style / Pipeline", loc="upper right", frameon=True,fontsize=9, title_fontsize=10)
        plt.tight_layout()
        plt.show()


def plot_loss_summary(csv_file: str, figurename):
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
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 5))
    plt.suptitle("Training & Validation Losses Across Pipelines (Averaged Over Seeds)")

    for pipeline in summary["pipeline"].unique():
        sub = summary[summary["pipeline"] == pipeline]
        ax1.plot(sub["epoch"], sub["train_mean"], label=pipeline, linewidth=2)
        # plt.fill_between(
        #     sub["epoch"],
        #     sub["train_mean"] - sub["train_std"],
        #     sub["train_mean"] + sub["train_std"],
        #     alpha=0.2
        # )
   
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.legend(title="Pipeline")
    ax1.grid(True, linestyle="--", alpha=0.6)
    # ax1.tight_layout()
    # ax1.show()

    # --- Validation Loss ---
    # plt.figure(figsize=(10, 6))
    for pipeline in summary["pipeline"].unique():
        sub = summary[summary["pipeline"] == pipeline]
        ax2.plot(sub["epoch"], sub["val_mean"], label=pipeline, linewidth=2)
        # plt.fill_between(
        #     sub["epoch"],
        #     sub["val_mean"] - sub["val_std"],
        #     sub["val_mean"] + sub["val_std"],
        #     alpha=0.2
        # )
    # ax2.set_title("Validation Loss Across Pipelines (Averaged Over Seeds)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Loss")
    ax2.legend(title="Pipeline")
    ax2.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(f"results/Images/{figurename}.png",  dpi=300, bbox_inches="tight")
    
    plt.show()


# ========================================
def plot_model_comparison(csv_files, labels, save_path=None):
    '''
    Args:
        csv_files: List of csv file paths.
        label: List of model names corresponding to each csv file.
    Output:
        (1) Training loss
        (2) Validation loss
            - The lowest validation loss for each model is highlighted with its value and epoch
        (3) Validation Dice 
            - The highest validation Dice score for each model is highlighted with its value and epoch

    Usage:
        csv_files = [
            "results/model comparison/base/all_pipelines_history.csv",
            "results/model comparison/optimized/all_pipelines_history.csv"
        ]
        labels = ["Baseline", "Optimized"]

        plot_model_comparison(csv_files, labels)
    '''
    plt.figure(figsize=(15,19))
    
    # --- Train Loss ---
    plt.subplot(3,1,1)
    for csv, label in zip(csv_files, labels):
        df = pd.read_csv(csv)
        plt.plot(df["epoch"], df["train_loss"], label=f"{label} - train")
    plt.title("Training Loss across models")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # --- Validation Loss ---
    plt.subplot(3,1,2)
    colors = plt.get_cmap("tab10").colors  
    for i, (csv, label) in enumerate(zip(csv_files, labels)):
        df = pd.read_csv(csv)
        color = colors[i % len(colors)]
        plt.plot(df["epoch"], df["val_loss"], color=color, label=f"{label} - val")

        min_idx = df["val_loss"].idxmin()
        best_epoch = df.loc[min_idx, "epoch"]
        best_val = df.loc[min_idx, "val_loss"]
        plt.scatter(best_epoch, best_val, color=color, zorder=5)
        plt.text(best_epoch-1.5, best_val+0.05, f"{best_val:.3f}\n(E{best_epoch})",
                 fontsize=8, color=color)
    
    plt.title("Validation Loss across models")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # --- Validation Dice ---
    plt.subplot(3,1,3)
    for i, (csv, label) in enumerate(zip(csv_files, labels)):
        df = pd.read_csv(csv)
        color = colors[i % len(colors)]
        plt.plot(df["epoch"], df["val_dice"], color=color, label=f"{label} - val")

        max_idx = df["val_dice"].idxmax()
        best_epoch = df.loc[max_idx, "epoch"]
        best_val = df.loc[max_idx, "val_dice"]
        plt.scatter(best_epoch, best_val, color=color, zorder=5)
        plt.text(best_epoch-1.5, best_val-0.06, f"{best_val:.3f}\n(E{best_epoch})",
                 fontsize=8, color=color)
    
    plt.title("Validation Dice across models")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Dice")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
#===========visualize prediction==============
def visualize_prediction_multiview(model, val_loader, device, slice_idxs=(60, 60, 60),title="Segmentation Results", save_path=True):
    """
    Visualize axial / coronal / sagittal slices together:
    (FLAIR MRI, Ground Truth, Prediction) for each view.
    """
    model.eval()
    imgs, masks = next(iter(val_loader))           
    imgs, masks = imgs.to(device), masks.to(device)

    with torch.no_grad():
        logits = model(imgs)                       
        preds  = torch.argmax(logits, dim=1)       
        gts    = torch.argmax(masks, dim=1)        

    # pick first sample
    img, gt, pr = imgs[0], gts[0], preds[0]  # img: (4,D,H,W), gt/pr: (D,H,W)

    axial_idx, coronal_idx, sagittal_idx = slice_idxs

    # slice extraction
    axial_img, axial_gt, axial_pr = img[0, axial_idx].cpu(), gt[axial_idx].cpu(), pr[axial_idx].cpu()
    coronal_img, coronal_gt, coronal_pr = img[0, :, coronal_idx, :].cpu(), gt[:, coronal_idx, :].cpu(), pr[:, coronal_idx, :].cpu()
    sagittal_img, sagittal_gt, sagittal_pr = img[0, :, :, sagittal_idx].cpu(), gt[:, :, sagittal_idx].cpu(), pr[:, :, sagittal_idx].cpu()

    # normalize MRI
    def normalize_img(x):
        x = x.numpy()
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    axial_img, coronal_img, sagittal_img = map(normalize_img, [axial_img, coronal_img, sagittal_img])

    # colormap
    cmap = ListedColormap(["black", "gold", "deepskyblue", "crimson"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    views = [
        ("Axial", axial_img, axial_gt, axial_pr),
        ("Coronal", coronal_img, coronal_gt, coronal_pr),
        ("Sagittal", sagittal_img, sagittal_gt, sagittal_pr)
    ]

    for i, (view, imgv, gt2d, pr2d) in enumerate(views):
        axes[i, 0].imshow(imgv, cmap="gray")
        axes[i, 0].set_title(f"{view} MRI (FLAIR)")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(imgv, cmap="gray")
        axes[i, 1].imshow(gt2d, cmap=cmap, norm=norm, alpha=0.6)
        axes[i, 1].set_title(f"{view} Ground Truth")
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(imgv, cmap="gray")
        axes[i, 2].imshow(pr2d, cmap=cmap, norm=norm, alpha=0.6)
        axes[i, 2].set_title(f"{view} Prediction")
        axes[i, 2].axis("off")
        
    # add legend
    legend_elements = [
        Patch(facecolor="black", label="Background", alpha=0.5),
        Patch(facecolor="gold", label="Non-enhancing core", alpha=0.5),
        Patch(facecolor="deepskyblue", label="Edema", alpha=0.5),
        Patch(facecolor="crimson", label="Enhancing tumor", alpha=0.5),
    ]
    fig.legend(handles=legend_elements, bbox_to_anchor=(1.05, 0.5), loc="center left")
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

    if save_path is True:
        save_path = f"{title}.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f" Figure saved to {save_path}")
    
