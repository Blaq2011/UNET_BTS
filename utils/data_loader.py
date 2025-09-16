import os, glob
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import random

import matplotlib.pyplot as plt



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










#####%%%%%%%%%%%%%%%%%%%%%%
# ------------------------------------
# Load Images and Masks
# ------------------------------------
def load_nifti(filepath):
    """Load NIfTI MRI file as numpy array"""
    img = nib.load(filepath)
    return img.get_fdata().astype(np.float32)
#####%%%%%%%%%%%%%%%%%%%%%%




# ------------------------------------
# Preprocessing utilities
# ------------------------------------
def clip_intensity(img, lower=0.5, upper=99.5):
    """Clip intensities to remove outliers"""
    low, high = np.percentile(img, [lower, upper])
    return np.clip(img, low, high)


def normalize(img):
    """Z-score normalization (ignores background)"""
    mask = img > 0
    if np.sum(mask) == 0:  # avoid NaN if image is empty
        return img
    mean = np.mean(img[mask])
    std = np.std(img[mask])
    return (img - mean) / (std + 1e-8)

def minmax_normalize(img):
    """Min-max normalization: scale nonzero foreground to [0,1]."""
    nz = img > 0
    if nz.sum() == 0:
        return img
    vmin = img[nz].min()
    vmax = img[nz].max()
    if abs(vmax - vmin) < 1e-6:   # avoid div by zero
        return img
    out = img.copy()
    out[nz] = (out[nz] - vmin) / (vmax - vmin)
    return out
  

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

def get_foreground(mods,eps=1e-6):
    """Union of non-zero voxels across modalities -> (H,W,D)bool."""
    stack = np.stack(mods,axis=0)
    return (np.abs(stack)>eps).any(axis=0)

def mask_slices(mask, margin=10):
    """Compute a single 3D slice tuple for cropping with safety margin."""
    coords = np.array(np.nonzero(mask))
    if coords.size == 0:
        return tuple(slice(0, s) for s in mask.shape)
    shape = np.array(mask.shape, dtype=int)
    mins = coords.min(axis=1) - margin
    maxs = coords.max(axis=1) + 1 + margin  # +1 to include max index
    mins = np.maximum(mins, 0)
    maxs = np.minimum(maxs, shape)
    return tuple(slice(int(mins[i]), int(maxs[i])) for i in range(3))


#####%%%%%%%%%%%%%%%%%%%%%%
def one_hot_encode(mask, num_classes=4):
    """
    Convert BraTS tumor mask into one-hot format.
    Original labels: 0=background, 1=edema, 2=non-enhancing, 4=enhancing
    Remapped labels: 0=background, 1=edema, 2=non-enhancing, 3=enhancing
    """
    mask = mask.astype(np.int32)
    mask[mask == 4] = 3  # remap label 4 -> 3
    return np.eye(num_classes)[mask]  # shape: (D,H,W,C)

#####%%%%%%%%%%%%%%%%%%%%%%

# ------------------------------------
# BraTS Dataset
# ------------------------------------
class BraTSDataset(Dataset):
    def __init__(self, image_paths, mask_paths, normalization, 
                 target_shape=(128,128,128), 
                 patch_size=(96,96,96), 
                 augment=True, 
                 num_classes=4,
                 crop_margin=10,
                 clip_percentiles=(0.5,99.5)):
        """
        image_paths: list of list, each [flair, t1, t1ce, t2]
        mask_paths: list of segmentation paths
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.normalization = normalization
        self.target_shape = target_shape
        self.patch_size = patch_size
        self.augment = augment
        self.num_classes = num_classes
        self.crop_margin = crop_margin
        self.clip_percentiles = clip_percentiles

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # --- Load modalities ---
        modalities = []
        for path in self.image_paths[idx]:
            img_array = load_nifti(path)
            modalities.append(img_array)

        # Stack into (C, D, H, W)
        img = np.stack(modalities, axis=0)

        # --- Load mask ---
        mask = load_nifti(self.mask_paths[idx])

        # --- Crop around tumor/brain ---
        fg = get_foreground(modalities)
        s  = mask_slices(fg, margin=self.crop_margin)
        modalities = [m[s] for m in modalities]
        mask = mask[s]

        # --- Resample ---
        img = np.stack(modalities, axis=0)
        img = resample_to_shape(img, self.target_shape, mode="trilinear")  # (4,D,H,W)
        mask = resample_to_shape(mask, self.target_shape, mode="nearest")  # (D,H,W)

        # --- Clip and normalization for each modality ---
        low, high= self.clip_percentiles
        for c in range(img.shape[0]):
          img[c] = clip_intensity(img[c], low, high)
          if self.normalization == "zscore":
                img[c] = normalize(img[c])      
          elif self.normalization == "minmax":
                img[c] = minmax_normalize(img[c])

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
            if random.random() > 0.5:  # flip z
                img_patch = np.flip(img_patch, axis=1).copy()
                mask_patch = np.flip(mask_patch, axis=0).copy()
            if random.random() > 0.5:  # flip y
                img_patch = np.flip(img_patch, axis=2).copy()
                mask_patch = np.flip(mask_patch, axis=1).copy()
            if random.random() > 0.5:  # flip x
                img_patch = np.flip(img_patch, axis=3).copy()
                mask_patch = np.flip(mask_patch, axis=2).copy()

        # --- Convert to torch tensors ---
        img_patch = torch.tensor(img_patch).float()                     # (4,d,h,w)
        mask_patch = torch.tensor(mask_patch).permute(3,0,1,2).float()  # (C,d,h,w)

        return img_patch, mask_patch












































def visualize_sample(img, mask, slice_idx=None):
    """
    Visualize one slice from a 3D BraTS sample.
    img: torch tensor (C,D,H,W)
    mask: torch tensor (C,D,H,W) one-hot
    slice_idx: which slice along depth (D) to show
    """
    img = img.numpy()
    mask = mask.numpy()
    
    C, D, H, W = img.shape
    
    # pick middle slice if none specified
    if slice_idx is None:
        slice_idx = D // 2

    # extract modalities at chosen slice
    modalities = [img[c, slice_idx, :, :] for c in range(C)]
    
    # extract segmentation mask (argmax from one-hot)
    mask_slice = mask[:, slice_idx, :, :].argmax(0)  # (H,W)

    fig, axes = plt.subplots(1, C+1, figsize=(15,5))
    titles = ["Flair", "T1", "T1ce", "T2", "Segmentation"]

    for i in range(C):
        axes[i].imshow(modalities[i], cmap="gray")
        axes[i].set_title(titles[i])
        axes[i].axis("off")

    axes[C].imshow(mask_slice, cmap="nipy_spectral")
    axes[C].set_title(titles[-1])
    axes[C].axis("off")

    plt.tight_layout()
    plt.show()
