import numpy as np
import torch

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

def compare_pipelines_dice(P1, P2, P3, num_samples=20, seed=42):
    """
    Compare P2 and P3 masks against P1 reference using Dice.
    Samples from the same patient index across datasets.
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(P1), size=num_samples, replace=False)

    dice_p2, dice_p3 = [], []

    for idx in indices:
        # get patches (what model sees)
        _, mask1 = P1[idx]  # (C,d,h,w)
        _, mask2 = P2[idx]
        _, mask3 = P3[idx]

        # convert one-hot to label map
        mask1 = mask1.argmax(0).numpy()
        mask2 = mask2.argmax(0).numpy()
        mask3 = mask3.argmax(0).numpy()

        dice_p2.append(dice_score(mask1, mask2))
        dice_p3.append(dice_score(mask1, mask3))

    return {
        "P2_vs_P1": (np.mean(dice_p2, axis=0), np.std(dice_p2, axis=0)),
        "P3_vs_P1": (np.mean(dice_p3, axis=0), np.std(dice_p3, axis=0)),
    }
