import torch
import numpy as np
from tqdm import tqdm
import time

# ----------------------
# Training Loop with Early Stopping
# ----------------------
def train_unet_baseline(
    model, train_loader, val_loader, optimizer, loss_fn, device, save_path,
    epochs, patience, scheduler=None 
):
    model = model.to(device)
    best_val_loss = np.inf
    patience_counter = 0

    history = {"train_loss": [], "val_loss": [], "lr": []}


    start_time = time.time()
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None

    for epoch in range(epochs):
        epoch_start = time.time()

        # Training
        model.train()
        train_loss = 0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = loss_fn(outputs, masks)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch+1}/{epochs}: "
            f"Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, "
            f"Time={epoch_time:.2f}s"
        )

        if scheduler is not None:
            scheduler.step(avg_val_loss)
            # Print LR change
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"--> LR adjusted to {current_lr:.2e}")

        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved at {save_path} (Val Loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    total_time = time.time() - start_time
    avg_epoch_time = total_time / (epoch + 1)
    gpu_mem = (
        torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
    )

    print(f"\n=== Training Summary ===")
    print(f"Total time: {total_time/60:.2f} min")
    print(f"Avg epoch time: {avg_epoch_time:.2f} s")
    if gpu_mem:
        print(f"Peak GPU memory: {gpu_mem:.2f} MB")

    return history, total_time, avg_epoch_time, gpu_mem


# ----------------------
# Experiment Runner
# ----------------------
def run_experiment(model, optimizer, loss_fn, train_loader, val_loader, pipeline_name, device, epochs, lr, patience,scheduler ):
    """
    Wrapper to train UNet on a given pipeline.
    Returns: path to best saved model + stats
    """
    save_path = f"models/unet_{pipeline_name}.pth"

    print(f"\n=== Training UNet on {pipeline_name} ===")
    history, total_time, avg_epoch_time, gpu_mem = train_unet_baseline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        save_path=save_path,
        epochs=epochs,
        patience=patience,
        scheduler = scheduler
    )

    return save_path, history, total_time, avg_epoch_time, gpu_mem
