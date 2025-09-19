import os, glob, random
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

#Function to collect image and mask file paths for our data loader 
def get_brats_filepaths(root_dir):
    """Collect BraTS2020 file paths grouped by patient"""
    patient_dirs = sorted(glob.glob(os.path.join(root_dir, "BraTS20_Training_*")))
    
    image_paths = []
    mask_paths = []

    for pdir in patient_dirs:
        patient_id = os.path.basename(pdir)

        # Four input modalities
        modalities = [
            os.path.join(pdir, f"{patient_id}_flair.nii"),
            os.path.join(pdir, f"{patient_id}_t1.nii"),
            os.path.join(pdir, f"{patient_id}_t1ce.nii"),
            os.path.join(pdir, f"{patient_id}_t2.nii")
        ]

        # Segmentation mask
        seg = os.path.join(pdir, f"{patient_id}_seg.nii")

        image_paths.append(modalities)
        mask_paths.append(seg)

    return image_paths, mask_paths

##Splitting data into train and val sets (critical before caching!) 
def split_brats_dataset(image_paths, mask_paths, val_size=0.2, seed=42):
    """
    Split BraTS training data into train/val sets (patient-wise).
    """
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths, mask_paths,
        test_size=val_size,
        random_state=seed,
        shuffle=True
    )
    return train_imgs, val_imgs, train_masks, val_masks



# Load Images and Masks
def load_nifti(filepath):
    """Load NIfTI MRI file as numpy array"""
    img = nib.load(filepath)
    return img.get_fdata().astype(np.float32)



###=====================Preprocessing utilities =======================
def clip_intensity(img, lower=0.5, upper=99.5):
    """Clip intensities to remove outliers"""
    low, high = np.percentile(img, [lower, upper])
    img = np.clip(img, low, high)
    return img


def normalize(img):
    """Z-score normalization (ignores background)"""
    mask = img > 0
    if np.sum(mask) == 0:  # avoid NaN if image is empty
        return img
    mean = np.mean(img[mask])
    std = np.std(img[mask])
    return (img - mean) / (std + 1e-8)


def resample_to_shape(img, target_shape=(128, 128, 128), mode="trilinear"):
    """
    Resample 3D (D,H,W) or 4D (C,D,H,W) image to target shape.
    mode: "trilinear" for images, "nearest" for masks
    """
    if img.ndim == 3:
        img_t = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
        if mode == "nearest":
            img_resampled = F.interpolate(img_t, size=target_shape, mode=mode)
        else:
            img_resampled = F.interpolate(img_t, size=target_shape, mode=mode, align_corners=False)
        return img_resampled.squeeze().numpy()

    elif img.ndim == 4:  # (C,D,H,W)
        resampled = []
        for c in range(img.shape[0]):
            channel_resized = resample_to_shape(img[c], target_shape, mode)
            resampled.append(channel_resized)
        return np.stack(resampled, axis=0)

    else:
        raise ValueError(f"Unsupported img shape {img.shape}, expected 3D or 4D.")


def crop_to_mask(img, mask, margin=10):
    """Crop around the nonzero region of mask (or brain)"""
    coords = np.array(np.nonzero(mask))
    if coords.size == 0:  # empty mask fallback
        return img, mask
    min_coords = np.maximum(coords.min(axis=1) - margin, 0)
    max_coords = np.minimum(coords.max(axis=1) + margin, img.shape)
    
    slices = [slice(min_coords[i], max_coords[i]) for i in range(3)]
    return img[slices[0], slices[1], slices[2]], mask[slices[0], slices[1], slices[2]]


def one_hot_encode(mask, num_classes=4):
    """
    Convert BraTS tumor mask into one-hot format.
    Original labels: 0=background, 1=edema, 2=non-enhancing, 4=enhancing
    Remapped labels: 0=background, 1=edema, 2=non-enhancing, 3=enhancing
    """
    mask = mask.astype(np.int32)
    mask[mask == 4] = 3  # remap label 4 -> 3
    return np.eye(num_classes)[mask]  # shape: (D,H,W,C)

#====================================================================================


# ===================================================================================
# Cache Builder
def build_cache(image_paths, mask_paths, 
                target_shape=(128,128,128), 
                patch_size=(96,96,96), 
                num_patches=8,
                out_dir="cache"):
    """
    Build cache for P2 (volumes) and P3 (patches).
    - P2: saves full preprocessed volumes
    - P3: saves multiple precomputed patches
    """
    os.makedirs(os.path.join(out_dir, "volumes"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "patches"), exist_ok=True)

    for i in tqdm(range(len(image_paths)), desc="Caching patients"):
        pid = os.path.basename(os.path.dirname(image_paths[i][0]))

        # --- Load modalities ---
        modalities = []
        for path in image_paths[i]:
            img = load_nifti(path)
            img = clip_intensity(img)
            img = normalize(img)
            modalities.append(img)
        img = np.stack(modalities, axis=0)  # (4,D,H,W)

        # --- Load mask ---
        mask = load_nifti(mask_paths[i])

        # --- Crop ---
        img_cropped = []
        for c in range(img.shape[0]):
            cropped_img, cropped_mask = crop_to_mask(img[c], mask)
            img_cropped.append(cropped_img)
        img = np.stack(img_cropped, axis=0)
        mask = cropped_mask

        # --- Resample ---
        img = resample_to_shape(img, target_shape, mode="trilinear")
        mask = resample_to_shape(mask, target_shape, mode="nearest")

        # Save for P2
        np.savez_compressed(
            os.path.join(out_dir, "volumes", f"{pid}.npz"),
            vol=img.astype(np.float32),
            seg=mask.astype(np.uint8)
        )

        # Save for P3
        patch_dir = os.path.join(out_dir, "patches", pid)
        os.makedirs(patch_dir, exist_ok=True)

        d, h, w = patch_size
        _, D, H, W = img.shape
        for j in range(num_patches):
            z = np.random.randint(0, D - d + 1) if D > d else 0
            y = np.random.randint(0, H - h + 1) if H > h else 0
            x = np.random.randint(0, W - w + 1) if W > w else 0

            img_patch = img[:, z:z+d, y:y+h, x:x+w]
            mask_patch = mask[z:z+d, y:y+h, x:x+w]
            mask_patch = one_hot_encode(mask_patch, num_classes=4)

            np.savez_compressed(
                os.path.join(patch_dir, f"{pid}_patch{j}.npz"),
                vol=img_patch.astype(np.float32),
                seg=mask_patch.astype(np.uint8)
            )

    print(f"✅ Cache built at: {out_dir}/volumes (P2) and {out_dir}/patches (P3)")


# Shared Augmentation Function
def apply_augmentations(img_patch, mask_patch):
    """
    Apply random augmentations to a patch.
    Safe for MRI (flips, rotations, intensity jitter, Gaussian noise).
    """
    # --- Flips ---
    if random.random() > 0.5: 
        img_patch, mask_patch = np.flip(img_patch, 1).copy(), np.flip(mask_patch, 0).copy()
    if random.random() > 0.5: 
        img_patch, mask_patch = np.flip(img_patch, 2).copy(), np.flip(mask_patch, 1).copy()
    if random.random() > 0.5: 
        img_patch, mask_patch = np.flip(img_patch, 3).copy(), np.flip(mask_patch, 2).copy()

    # --- 90-degree rotations (in-plane) ---
    if random.random() > 0.5:
        k = random.choice([1, 2, 3])  # 90, 180, 270 deg
        img_patch = np.rot90(img_patch, k, axes=(2,3)).copy()  # rotate along (y,x)
        mask_patch = np.rot90(mask_patch, k, axes=(0,1)).copy()

    # --- Intensity scaling (brightness/contrast) ---
    if random.random() > 0.5:
        scale = random.uniform(0.9, 1.1)
        shift = random.uniform(-0.1, 0.1)
        img_patch = img_patch * scale + shift

    # --- Gaussian noise ---
    if random.random() > 0.5:
        noise = np.random.normal(0, 0.01, img_patch.shape)
        img_patch = img_patch + noise

    return img_patch, mask_patch


#### ========  Dataset Classes - Pipeline 1
class BraTSDatasetP1(Dataset): #P1 : on the fly
    def __init__(self, image_paths, mask_paths, 
                 target_shape=(128,128,128), 
                 patch_size=(96,96,96), 
                 augment=True, 
                 num_classes=4):
        """
        image_paths: list of list, each [flair, t1, t1ce, t2]
        mask_paths: list of segmentation paths
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.target_shape = target_shape
        self.patch_size = patch_size
        self.augment = augment
        self.num_classes = num_classes

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # --- Load modalities ---
        modalities = []
        for path in self.image_paths[idx]:
            img_array = load_nifti(path)
            img_array = clip_intensity(img_array)
            img_array = normalize(img_array)
            modalities.append(img_array)

        # Stack into (C, D, H, W)
        img = np.stack(modalities, axis=0)

        # --- Load mask ---
        mask = load_nifti(self.mask_paths[idx])

        # --- Crop around tumor/brain ---
        img_cropped = []
        for c in range(img.shape[0]):
            cropped_img, cropped_mask = crop_to_mask(img[c], mask)
            img_cropped.append(cropped_img)
        img = np.stack(img_cropped, axis=0)
        mask = cropped_mask

        # --- Resample ---
        img = resample_to_shape(img, self.target_shape, mode="trilinear")  # (4,D,H,W)
        mask = resample_to_shape(mask, self.target_shape, mode="nearest")  # (D,H,W)

        # --- Extract random patch ---
        d, h, w = self.patch_size
        _, D, H, W = img.shape
        z = random.randint(0, D - d) if D > d else 0
        y = random.randint(0, H - h) if H > h else 0
        x = random.randint(0, W - w) if W > w else 0

        img_patch = img[:, z:z+d, y:y+h, x:x+w]   # (4,d,h,w)
        mask_patch = mask[z:z+d, y:y+h, x:x+w]    # (d,h,w)

        # --- One-hot encode mask ---
        mask_patch = one_hot_encode(mask_patch, self.num_classes)  # (d,h,w,C)

        # --- Data augmentation ---
        if self.augment:
            img_patch, mask_patch = apply_augmentations(img_patch, mask_patch)


        # --- Convert to torch tensors ---
        img_patch = torch.tensor(img_patch).float()                     # (4,d,h,w)
        mask_patch = torch.tensor(mask_patch).permute(3,0,1,2).float()  # (C,d,h,w)

        return img_patch, mask_patch

# Dataset Classes - Pipeline 2
class BraTSDatasetP2(Dataset):  # P2: cached volumes
    def __init__(self, cache_dir, patient_ids, patch_size=(96,96,96), augment=True, num_classes=4):
        self.cache_dir = cache_dir
        self.patient_ids = patient_ids
        self.patch_size = patch_size
        self.augment = augment
        self.num_classes = num_classes

    def __len__(self): 
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        data = np.load(os.path.join(self.cache_dir, f"{pid}.npz"))
        img, mask = data["vol"], data["seg"]

        d,h,w = self.patch_size
        _,D,H,W = img.shape
        z,y,x = [random.randint(0, dim-size) if dim>size else 0 
                 for dim,size in zip((D,H,W),(d,h,w))]
        img_patch = img[:, z:z+d, y:y+h, x:x+w]
        mask_patch = mask[z:z+d, y:y+h, x:x+w]
        mask_patch = one_hot_encode(mask_patch, self.num_classes)

        # --- Data augmentation ---
        if self.augment:
            img_patch, mask_patch = apply_augmentations(img_patch, mask_patch)

                # --- Convert to torch tensors ---
        img_patch = torch.tensor(img_patch).float()                     # (4,d,h,w)
        mask_patch = torch.tensor(mask_patch).permute(3,0,1,2).float()  # (C,d,h,w)
        
        return img_patch, mask_patch

# Dataset Classes - Pipeline 3
class BraTSDatasetP3(Dataset):  # P3: cached patches
    def __init__(self, cache_dir, patient_ids, augment=True, num_classes=4):
        self.cache_dir = cache_dir
        self.patient_ids = patient_ids
        self.augment = augment
        self.num_classes = num_classes

    def __len__(self): return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        patch_files = glob.glob(os.path.join(self.cache_dir, pid, "*.npz"))
        patch_file = random.choice(patch_files)
        data = np.load(patch_file)
        img_patch, mask_patch = data["vol"], data["seg"]

        # --- Data augmentation ---
        if self.augment:
            img_patch, mask_patch = apply_augmentations(img_patch, mask_patch)

                # --- Convert to torch tensors ---
        img_patch = torch.tensor(img_patch).float()                     # (4,d,h,w)
        mask_patch = torch.tensor(mask_patch).permute(3,0,1,2).float()  # (C,d,h,w)

        return img_patch, mask_patch


# ===================================================================================



# ===================================================================================
#Visualizing the different pipelines
def visualize_patient_consistency(P1, P2, P3, patient_idx=0, slice_axis=0, deterministic=False):
    """
    Visualize dataset outputs for a given patient index.
    Each row = pipeline (P1, P2, P3).
    Each column = modality (FLAIR, T1, T1CE, T2, Mask).
    Shows exactly what the model sees (post-preprocessing).
    
    slice_axis: 0=axial(z), 1=coronal(y), 2=sagittal(x)
    deterministic: if True, disables randomness in patch/aug selection
                   (center patch, no augmentation) for reproducibility.
    """

    datasets = [P1, P2, P3]
    row_labels = ["P1 (on-the-fly)", "P2 (cached vol)", "P3 (cached patches)"]
    col_titles = ["FLAIR", "T1", "T1CE", "T2", "Mask"]

    rows = []
    for ds in datasets:
        if deterministic:
            # temporarily disable augmentation for consistency
            aug_state = getattr(ds, "augment", None)
            if aug_state is not None:
                ds.augment = False

            # force center patch if patch_size is defined
            if hasattr(ds, "patch_size"):
                d, h, w = ds.patch_size
                img_patch, mask_patch = ds[patient_idx]
                img_patch, mask_patch = img_patch.numpy(), mask_patch.numpy()
                mask_patch = np.argmax(mask_patch, axis=0)
            else:
                img_patch, mask_patch = ds[patient_idx]
                img_patch, mask_patch = img_patch.numpy(), mask_patch.numpy()
                mask_patch = np.argmax(mask_patch, axis=0)

            # restore augmentation state
            if aug_state is not None:
                ds.augment = aug_state
        else:
            # normal pipeline output (with randomness/aug)
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
    mode = "Deterministic (center, no aug)" if deterministic else "Random (as in training)"
    fig.suptitle(f"Patient {patient_idx} — {mode}, axis={slice_axis}", fontsize=16)

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
        fig.text(0.02, y_center, label, va="center", ha="left",
                 rotation=90, fontsize=12, fontweight="bold")

    out_path = f"results/images/patient-{patient_idx}_{'det' if deterministic else 'rand'}_consistency.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.show()




