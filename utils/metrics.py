import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure



# ----------------------
# Dice Score
# ----------------------
def dice_score(mask1, mask2, num_classes=4):
    """
    Compute Dice per class between two label maps.
    mask1, mask2: numpy arrays (d,h,w) with int labels
    """
    scores = []
    for c in range(num_classes):
        m1 = (mask1 == c).astype(np.uint8)
        m2 = (mask2 == c).astype(np.uint8)
        inter = np.sum(m1 * m2)
        denom = np.sum(m1) + np.sum(m2)
        dice = (2 * inter) / denom if denom > 0 else 1.0
        scores.append(dice)
    return np.array(scores)


# ----------------------
# Pipeline Comparison (Dice)
# ----------------------
def compare_pipelines_dice(P1, P2, P3, num_samples=20, seed=42):
    """
    Compare P2 and P3 masks against P1 reference using Dice.
    Samples from the same patient index across datasets.
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(P1), size=num_samples, replace=False)

    dice_p2, dice_p3 = [], []

    for idx in indices:
        _, mask1 = P1[idx]
        _, mask2 = P2[idx]
        _, mask3 = P3[idx]

        mask1 = mask1.argmax(0).numpy()
        mask2 = mask2.argmax(0).numpy()
        mask3 = mask3.argmax(0).numpy()

        dice_p2.append(dice_score(mask1, mask2))
        dice_p3.append(dice_score(mask1, mask3))

    return {
        "P2_vs_P1": (np.mean(dice_p2, axis=0), np.std(dice_p2, axis=0)),
        "P3_vs_P1": (np.mean(dice_p3, axis=0), np.std(dice_p3, axis=0)),
    }





#####From here --- PART ---  EVANS


# ----------------------
# Dice Loss

def dice_loss(pred, target, smooth=1e-5):
    """
    pred: (B, C, D, H, W) logits
    target: (B, C, D, H, W) one-hot
    Returns average Dice loss over batch and classes.
    """
    # apply softmax over classes
    pred = torch.softmax(pred, dim=1)
    B, C = pred.shape[:2]
    
    # flatten spatial dims: (B, C, N)
    pred_flat = pred.view(B, C, -1)
    target_flat = target.view(B, C, -1)
    
    # intersection and cardinality per sample and class
    intersection = (pred_flat * target_flat).sum(dim=2)
    cardinality = pred_flat.sum(dim=2) + target_flat.sum(dim=2)
    
    # Dice score per sample and class
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth)
    
    # Dice loss = 1 − Dice score, then average
    dice_loss_per_class = 1.0 - dice_score
    return dice_loss_per_class.mean()

class DiceCELoss(nn.Module):
    def __init__(self, ce_weight=None, alpha=0.5):
        """
        ce_weight: tensor of shape (C,) for class weighting in CrossEntropy
        alpha: weighting factor between CE and Dice (0 = only Dice, 1 = only CE)
        """
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight, reduction='mean')
        self.alpha = alpha

    def forward(self, pred, target_onehot):
        """
        pred: (B, C, D, H, W) logits
        target_onehot: (B, C, D, H, W) one-hot encoding
        """
        # CE expects class indices of shape (B, D, H, W)
        target_idx = target_onehot.argmax(dim=1)
        loss_ce = self.ce(pred, target_idx)
        loss_dice = dice_loss(pred, target_onehot)
        # balanced sum
        return self.alpha * loss_ce + (1 - self.alpha) * loss_dice


# BRATs standard dice (WT, TC, ET)

def dice_binary(a, b, eps=1e-5):
    inter = np.sum((a > 0) & (b > 0))
    denom = np.sum(a > 0) + np.sum(b > 0)
    return (2.0 * inter + eps) / (denom + eps)

def brats_regions_from_labels(lbl):
    """
    lbl: (D,H,W) int labels {0,1,2,3}
    Returns binary masks for WT, TC, ET.
    WT = 1|2|3; TC = 2|3; ET = 3
    """
    wt = (lbl == 1) | (lbl == 2) | (lbl == 3) # Non-Enh (1) + Edema (2) + Enh T (3)
    tc = (lbl == 1) | (lbl == 3) # Non-Enh(1) + Enh T (3)
    et = (lbl == 3)  # Enh T (3)
    # print("ET voxels in GT:", np.count_nonzero(lbl == 3))
    return wt.astype(np.uint8), tc.astype(np.uint8), et.astype(np.uint8)

def dice_wt_tc_et(pred_labels, gt_labels):
    """
    pred_labels, gt_labels: (D,H,W) int maps
    Returns a np.array([WT, TC, ET]) dice.
    """
    gt_wt, gt_tc, gt_et = brats_regions_from_labels(gt_labels)
    pr_wt, pr_tc, pr_et = brats_regions_from_labels(pred_labels)
    
    return np.array([
        dice_binary(pr_wt, gt_wt),
        dice_binary(pr_tc, gt_tc),
        dice_binary(pr_et, gt_et)
    ])
    

###Calculating HD95
'''
HD95 measures boundary distance between predicted and ground-truth segmentations.
Instead of the maximum Hausdorff distance (sensitive to outliers), 
HD95 computes the 95th percentile of distances between the two boundaries.
Lower = better.
'''


def surface_points(mask):
    """Return binary surface mask for a 3D binary volume."""
    struct = generate_binary_structure(3, 1)
    eroded = binary_erosion(mask, structure=struct, border_value=0)
    return mask ^ eroded

def hd95(pred, gt):
    """
    Compute 95th percentile Hausdorff Distance (HD95) between two binary masks.
    Memory-efficient version using distance transforms.
    Args:
        pred: (D,H,W) binary prediction
        gt:   (D,H,W) binary ground truth
    Returns:
        hd95 distance (float, in voxels)
    """
    assert pred.shape == gt.shape, "checks Shape mismatch between prediction and ground truth"
    
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    if pred.sum() == 0 or gt.sum() == 0:
        # print("Warning: One of the masks is empty.")
        return np.nan

    # Get surfaces
    pred_surface = surface_points(pred)
    gt_surface   = surface_points(gt)

    # Distance transforms
    dt_pred = distance_transform_edt(~pred_surface)
    dt_gt   = distance_transform_edt(~gt_surface)

    # Directed distances
    dists_pred_to_gt = dt_gt[pred_surface]
    dists_gt_to_pred = dt_pred[gt_surface]

    all_dists = np.concatenate([dists_pred_to_gt, dists_gt_to_pred])
    return np.percentile(all_dists, 95)


def hd95_wt_tc_et(pred_labels, gt_labels):
    """
    pred_labels, gt_labels: (D,H,W) int maps
    Returns a np.array([WT, TC, ET]) HD95 distances.
    """
    gt_wt, gt_tc, gt_et = brats_regions_from_labels(gt_labels)
    pred_wt, pred_tc, pred_et = brats_regions_from_labels(pred_labels)
    
    return np.array([
        hd95(pred_wt, gt_wt),
        hd95(pred_tc, gt_tc),
        hd95(pred_et, gt_et)
    ])
    

