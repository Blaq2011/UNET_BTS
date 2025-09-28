import torch
import numpy as np
from tqdm import tqdm
import time
from utils.metrics import dice_score, dice_wt_tc_et, hd95_wt_tc_et
import gc

# ----------------------
# Training Loop with Early Stopping (toggle option)
# ----------------------


def train_unet(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    save_path,
    epochs,
    patience,
    scheduler=None,
    early_stopping=False
):
    """
    Train a 3D UNet, track per‐epoch loss & dice (overall + per‐class),
    then compute final BraTS WT/TC/ET Dice and HD95 on the best‐saved model.
    Returns a single `history` dict containing:
      - train_loss                (list of floats)
      - val_loss                  (list of floats)
      - val_dice                  (list of floats: mean per‐class dice each epoch)
      - val_dice_per_class        (list of 4‐floats lists each epoch)
      - dice_brats_mean/std       (length‐3 lists for WT/TC/ET)
      - hd95_brats_mean/std       (length‐3 lists for WT/TC/ET)
      - dice_class_mean/std       (length‐4 lists for Background, NonEn, Edema, Enh)
      - time                      (total_time, avg_epoch_time, peak_gpu_mem_MB)
    """
    model = model.to(device)
    best_val = np.inf
    wait = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_dice_per_class": []
    }

    start = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # TRAIN / VALIDATION LOOPS
    for epoch in range(1, epochs + 1):
        # — TRAIN —
        model.train()
        running_train = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            if isinstance(outputs, (list, tuple)):
                n_heads = len(outputs)
                weights = [1.0 / (2**i) for i in range(n_heads)]
                loss = sum(w * loss_fn(o, masks) for w, o in zip(weights, outputs))
                main_out = outputs[0]
            else:
                loss = loss_fn(outputs, masks)
                main_out = outputs
            loss.backward()
            optimizer.step()
            running_train += loss.item()

        avg_train = running_train / len(train_loader)
        history["train_loss"].append(avg_train)

        # — VALIDATE —
        model.eval()
        running_val = 0.0
        dice_scores_pc = []
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"[ Val ] Epoch {epoch}/{epochs}"):
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                if isinstance(outputs, (list, tuple)):
                    loss = sum(w * loss_fn(o, masks) for w, o in zip(weights, outputs))
                    main_out = outputs[0]
                else:
                    loss = loss_fn(outputs, masks)
                    main_out = outputs
                running_val += loss.item()

                preds = torch.argmax(main_out, dim=1)
                gts   = masks.argmax(dim=1)
                for p, t in zip(preds, gts):
                    pcs = dice_score(
                        t.cpu().numpy(),
                        p.cpu().numpy(),
                        num_classes=4
                    )
                    dice_scores_pc.append(pcs)

        arr_pc           = np.stack(dice_scores_pc, axis=0)  # (N_samples, 4)
        mean_pc          = arr_pc.mean(axis=0)              # length‐4
        overall_val_dice = float(mean_pc.mean())            # scalar

        avg_val = running_val / len(val_loader)
        history["val_loss"].append(avg_val)
        history["val_dice_per_class"].append(mean_pc.tolist())
        history["val_dice"].append(overall_val_dice)

        if scheduler:
            scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            wait = 0
            torch.save(model.state_dict(), save_path)
        else:
            wait += 1
            if early_stopping and wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(
            f"Epoch {epoch}/{epochs}  "
            f"train_loss={avg_train:.4f}  "
            f"val_loss={avg_val:.4f}  "
            f"val_dice={overall_val_dice:.4f}"
        )

    # — FINAL BraTS WT/TC/ET Dice & HD95 on best model —
    model.load_state_dict(torch.load(save_path))
    model.eval()
    all_bdice = []
    all_bhd95 = []
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            logits  = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            preds   = torch.argmax(logits, dim=1).cpu().numpy()
            gts     = masks.argmax(dim=1).cpu().numpy()
            for p, t in zip(preds, gts):
                all_bdice.append(dice_wt_tc_et(p, t))
                all_bhd95.append(hd95_wt_tc_et(p, t))

    arr_dice = np.stack(all_bdice, axis=0)  # (N_samples, 3)
    arr_hd95 = np.stack(all_bhd95, axis=0)  # (N_samples, 3)

    history["dice_brats_mean"] = arr_dice.mean(axis=0).tolist()
    history["dice_brats_std"]  = arr_dice.std(axis=0).tolist()
    history["hd95_brats_mean"] = arr_hd95.mean(axis=0).tolist()
    history["hd95_brats_std"]  = arr_hd95.std(axis=0).tolist()

    # — FINAL per‐class Dice on best model —
    all_pc = []
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            logits  = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            preds   = torch.argmax(logits, dim=1).cpu().numpy()
            gts     = masks.argmax(dim=1).cpu().numpy()
            for p, t in zip(preds, gts):
                all_pc.append(dice_score(t, p, num_classes=4))

    arr_pc_final               = np.stack(all_pc, axis=0)  # (N_samples, 4)
    history["dice_class_mean"] = arr_pc_final.mean(axis=0).tolist()
    history["dice_class_std"]  = arr_pc_final.std(axis=0).tolist()

    # timing & memory
    total_time     = time.time() - start
    avg_epoch_time = total_time / len(history["train_loss"])
    peak_mem       = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else None
    )
    history["time"] = (total_time, avg_epoch_time, peak_mem)

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return history


# ----------------------
# Experiment Runner
# ----------------------
def run_experiment(
    model,
    optimizer,
    loss_fn,
    train_loader,
    val_loader,
    pipeline_name,
    device,
    epochs,
    lr,
    patience,
    scheduler=None,
    early_stopping=False
):
    """
    Wrapper to train UNet on a given pipeline.
    Returns:
      save_path (str),
      history (dict of lists & metrics),
      total_time (float, seconds),
      avg_epoch_time (float, seconds),
      gpu_mem (float, MB)
    """
    save_path = f"models/unet_{pipeline_name}.pth"
    print(f"\n=== Training UNet on {pipeline_name} ===")

    # train_unet  returns a single `history` dict,
    # whose `history["time"]` is (total_time, avg_epoch_time, gpu_mem)
    history = train_unet(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        save_path=save_path,
        epochs=epochs,
        patience=patience,
        scheduler=scheduler,
        early_stopping=early_stopping
    )

    # extract timing/memory from history
    total_time, avg_epoch_time, gpu_mem = history.pop("time")

    return save_path, history, total_time, avg_epoch_time, gpu_mem