
import os
import torch
import numpy as np
import pandas as pd

from utils.seeding import set_global_seed
from utils.train_unet import run_experiment


def run_train_eval(
    seeds: list,
    pipelines: dict,
    model_fn,
    loss_fn_fn,
    optimizer_fn,
    scheduler_fn=None,
    early_stopping = True,
    run_experiment_fn=run_experiment,
    epochs: int = 30,
    patience: int = 10,
    lr: float = 1e-4,
    device: torch.device = None,
    results_dir: str = "results"
):
    """
    Train & evaluate multiple UNet pipelines across several seeds.

    Parameters:
      seeds           : list of int random seeds
      pipelines       : dict mapping pipeline_name - > (train_loader, val_loader)
      model_fn        : callable --> new model instance (e.g. lambda: UNet3D(...))
      loss_fn_fn      : callable --> new loss instance (e.g. DiceCELoss)
      optimizer_fn    : callable(model) --> optimizer (e.g. lambda m: AdamW(m.parameters(), lr=lr))
      scheduler_fn    : optional callable(optimizer) --> scheduler or None
      run_experiment_fn: function to train+eval a single run
      epochs          : number of epochs per run
      patience        : early‐stop patience on val_loss
      lr              : learning rate (for display/passing)
      device          : torch.device
      results_dir     : where to save CSVs

    Returns:
      df_all_hist : pandas.DataFrame of all per‐epoch histories
      df_summary  : pandas.DataFrame of aggregated final metrics
    """
    os.makedirs(results_dir, exist_ok=True)
    all_histories = []
    all_results = {name: [] for name in pipelines.keys()}

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        set_global_seed(seed)

        for name, (train_loader, val_loader) in pipelines.items():
            print(f"\n--- Pipeline {name} ---")
            model     = model_fn().to(device)
            loss_fn   = loss_fn_fn()
            optimizer = optimizer_fn(model)
            scheduler = scheduler_fn(optimizer) if scheduler_fn else None

            # run training + evaluation
            ckpt, hist, total_time, avg_epoch_time, peak_mem = run_experiment_fn(
                model, optimizer, loss_fn,
                train_loader, val_loader,
                f"{name}_s{seed}", device,
                epochs, lr, patience,
                scheduler=scheduler,
                early_stopping=early_stopping
            )
            
            print(f"\n=== Training Summary ===")
            print(f"Total time: {total_time/60:.2f} min")
            print(f"Avg epoch time: {avg_epoch_time:.2f} s")
            print(f"Peak GPU memory: {peak_mem:.2f} MB")

            # build per‐epoch history DataFrame
            n = len(hist["train_loss"])
            df_hist = pd.DataFrame({
                "seed":               [seed] * n,
                "pipeline":           [name] * n,
                "epoch":              list(range(1, n+1)),
                "train_loss":         hist["train_loss"],
                "val_loss":           hist["val_loss"],
                "val_dice":           hist["val_dice"],
                "dice_bg":            [v[0] for v in hist["val_dice_per_class"]],
                "dice_non_enh":       [v[1] for v in hist["val_dice_per_class"]],
                "dice_edema":         [v[2] for v in hist["val_dice_per_class"]],
                "dice_enhancing":     [v[3] for v in hist["val_dice_per_class"]],
            })
            path_hist = os.path.join(results_dir, f"{name}_history_{seed}.csv")
            df_hist.to_csv(path_hist, index=False)
            print(f"Saved history → {path_hist}")
            all_histories.append(df_hist)

            # collect final metrics
            all_results[name].append({
                "dice_class_mean":   hist["dice_class_mean"],
                "dice_class_std":    hist["dice_class_std"],
                "dice_brats_mean":   hist["dice_brats_mean"],
                "dice_brats_std":    hist["dice_brats_std"],
                "hd95_brats_mean":   hist["hd95_brats_mean"],
                "hd95_brats_std":    hist["hd95_brats_std"],
                "time":              (total_time, avg_epoch_time, peak_mem)
            })

    # combine & save all histories
    df_all_hist = pd.concat(all_histories, ignore_index=True)
    path_all_hist = os.path.join(results_dir, "all_pipelines_history.csv")
    df_all_hist.to_csv(path_all_hist, index=False)
    print(f"\nSaved all histories → {path_all_hist}")

    # aggregate final metrics across seeds
    print("\n=== Final aggregated results ===")
    summary_rows = []
    for name, res_list in all_results.items():
        pc_arr    = np.stack([r["dice_class_mean"] for r in res_list], axis=0)
        mean_pc   = pc_arr.mean(axis=0)
        std_pc    = pc_arr.std(axis=0)

        db        = np.stack([r["dice_brats_mean"] for r in res_list], axis=0)
        hb        = np.stack([r["hd95_brats_mean"] for r in res_list], axis=0)
        mean_dice = db.mean(axis=0)
        std_dice  = db.std(axis=0)
        mean_hd95 = hb.mean(axis=0)
        std_hd95  = hb.std(axis=0)

        print(f"\nPipeline {name}:")
        print("  Per-class Dice (BG, NonEn, Edema, Enh):")
        for cls, m, s in zip(["BG","NonEn","Edema","Enh"], mean_pc, std_pc):
            print(f"    {cls:8s}: {m:.3f} ± {s:.3f}")

        print("  Dice (WT, TC, ET):")
        for r, m, s in zip(["WT","TC","ET"], mean_dice, std_dice):
            print(f"    {r}: {m:.3f} ± {s:.3f}")

        print("  HD95 (WT, TC, ET):")
        for r, m, s in zip(["WT","TC","ET"], mean_hd95, std_hd95):
            print(f"    {r}: {m:.3f} ± {s:.3f}")

        summary_rows.append({
            "pipeline":         name,
            "dice_WT_mean":     mean_dice[0], "dice_WT_std":     std_dice[0],
            "dice_TC_mean":     mean_dice[1], "dice_TC_std":     std_dice[1],
            "dice_ET_mean":     mean_dice[2], "dice_ET_std":     std_dice[2],
            "hd95_WT_mean":     mean_hd95[0], "hd95_WT_std":     std_hd95[0],
            "hd95_TC_mean":     mean_hd95[1], "hd95_TC_std":     std_hd95[1],
            "hd95_ET_mean":     mean_hd95[2], "hd95_ET_std":     std_hd95[2],
        })

    df_summary = pd.DataFrame(summary_rows)
    path_summary = os.path.join(results_dir, "all_results.csv")
    df_summary.to_csv(path_summary, index=False)
    print(f"Saved summary → {path_summary}")

    return df_all_hist, df_summary