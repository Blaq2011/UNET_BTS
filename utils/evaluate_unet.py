import torch
import numpy as np
from utils.unet import UNet3D
from utils.metrics import dice_score, dice_wt_tc_et
import matplotlib.pyplot as plt

def evaluate_model(model, val_loader, device, class_names):
    """
    Compute both per-class Dice (debug) and BraTS composite Dice (WT/TC/ET).
    """
    model.eval()
    dices_per_class = []
    dices_brats = []  # WT, TC, ET

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)       # (B,4,d,h,w), (B,4,d,h,w)
            logits = model(imgs)                                   # (B,4,d,h,w)
            preds = torch.argmax(logits, dim=1).cpu().numpy()      # (B,d,h,w)
            gts   = masks.argmax(dim=1).cpu().numpy()              # (B,d,h,w)

            for p, t in zip(preds, gts):
                # Per-class (Background, Edema, Non-enh, Enh)
                dices_per_class.append(dice_score(t, p, num_classes=len(class_names)))
                # BraTS-style WT/TC/ET
                dices_brats.append(dice_wt_tc_et(p, t))

    dices_per_class = np.array(dices_per_class)  # (N,4)
    dices_brats     = np.array(dices_brats)      # (N,3)

    mean_pc, std_pc = dices_per_class.mean(axis=0), dices_per_class.std(axis=0)
    mean_b, std_b   = dices_brats.mean(axis=0),  dices_brats.std(axis=0)

    # Debug printout
    print("\nPer-class Dice (debugging and pipeline selection):")
    for i, cls in enumerate(class_names):
        print(f"  {cls:15s}: {mean_pc[i]:.3f} ± {std_pc[i]:.3f}")

    # Report printout
    print("\nBraTS region Dice (Model comparison , BRATS format):")
    for name, m, s in zip(["WT", "TC", "ET"], mean_b, std_b):
        print(f"  {name:2s}: {m:.3f} ± {s:.3f}")

    return {
        "dice_class": (mean_pc, std_pc),   # optional, for debugging
        "dice_brats": (mean_b, std_b),     # WT/TC/ET — what you report
    }


def eval_experiment(model, val_loader, model_path, pipeline_name, device):
    """
    Load a saved model checkpoint and evaluate.
    """
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    class_names = ["Background", "Edema", "Non-enhancing", "Enhancing"]

    print(f"\n=== Results for {pipeline_name} ===")
    return evaluate_model(model, val_loader, device, class_names=class_names)


