import torch
import numpy as np
from utils.metrics import dice_score, dice_wt_tc_et, hd95_wt_tc_et
import matplotlib.pyplot as plt

def evaluate_model(model, val_loader, device, class_names):
    """
    Compute both per-class Dice (debug) and BraTS composite Dice (WT/TC/ET)
    as well as HD95(WT/TC/ET). Handles both single-head and multi-head UNets.
    """
    model.eval()
    dices_per_class = []  # Background, Non-enh, Edema, Enh
    dices_brats     = []  # WT, TC, ET
    hd95_brats      = []  # WT, TC, ET

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)  # (B,4,d,h,w), (B,4,d,h,w)

            outputs = model(imgs)
            # If model returns a list of deep-supervision heads, take the main head
            if isinstance(outputs, (list, tuple)):
                logits = outputs[0]
            else:
                logits = outputs

            # logits: (B,4,d,h,w)
            preds = torch.argmax(logits, dim=1).cpu().numpy()  # (B,d,h,w)
            gts   = masks.argmax(dim=1).cpu().numpy()         # (B,d,h,w)

            for p, t in zip(preds, gts):
                # Per-class Dice (Background, Non-enh, Edema, Enh)
                dices_per_class.append(dice_score(t, p, num_classes=len(class_names)))
                # BraTS-style Dice WT/TC/ET
                dices_brats.append(dice_wt_tc_et(p, t))
                # BraTS-style HD95 WT/TC/ET
                hd95_brats.append(hd95_wt_tc_et(p, t))

    dices_per_class = np.array(dices_per_class)  # (N,4)
    dices_brats     = np.array(dices_brats)      # (N,3)
    hd95_brats      = np.array(hd95_brats)       # (N,3)

    mean_pc = np.nanmean(dices_per_class, axis=0)
    std_pc  = np.nanstd(dices_per_class, axis=0)
    mean_b  = np.nanmean(dices_brats, axis=0)
    std_b   = np.nanstd(dices_brats, axis=0)
    mean_hd = np.nanmean(hd95_brats, axis=0)
    std_hd  = np.nanstd(hd95_brats, axis=0)

    # Debug printout: per-class Dice
    print("\nPer-class Dice (debugging and pipeline selection):")
    for i, cls in enumerate(class_names):
        print(f"  {cls:15s}: {mean_pc[i]:.3f} ± {std_pc[i]:.3f}")

    # Report: BraTS region Dice
    print("\nBraTS region Dice (Model comparison, BRATS format):")
    for name, m, s in zip(["WT", "TC", "ET"], mean_b, std_b):
        print(f"  {name:2s}: {m:.3f} ± {s:.3f}")

    # Report: BraTS HD95
    print("\nBraTS HD95 (Model comparison, BRATS format):")
    for name, m, s in zip(["WT", "TC", "ET"], mean_hd, std_hd):
        print(f"  {name:2s}: {m:.3f} ± {s:.3f}")

    return {
        "dice_class": (mean_pc, std_pc),
        "dice_brats": (mean_b, std_b),
        "hd95_brats": (mean_hd, std_hd),
    }


def eval_experiment(model, val_loader, model_path, pipeline_name, device):
    """
    Load a saved model checkpoint and evaluate.
    """
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    class_names = ["Background", "Non-enhancing", "Edema", "Enhancing"]

    print(f"\n=== Results for {pipeline_name} ===")
    return evaluate_model(model, val_loader, device, class_names=class_names)


