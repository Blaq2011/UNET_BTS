
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", context="talk")


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
