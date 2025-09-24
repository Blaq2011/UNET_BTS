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

        # building Four input modalities paths
        modalities = [
            os.path.join(pdir, f"{patient_id}_flair.nii"),
            os.path.join(pdir, f"{patient_id}_t1.nii"),
            os.path.join(pdir, f"{patient_id}_t1ce.nii"),
            os.path.join(pdir, f"{patient_id}_t2.nii")
        ]

        # building Segmentation mask paths
        seg = os.path.join(pdir, f"{patient_id}_seg.nii")

        image_paths.append(modalities)
        mask_paths.append(seg)

    return image_paths, mask_paths #returns 4-element lists of modality paths and a list of single mask paths

##Splitting data into train and val sets (critical before caching!) 
def split_brats_dataset(image_paths, mask_paths, val_size=0.2, seed=42):
    """
    Split BraTS training data into train/val sets (patient-wise).
    80-20(train-val) split
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
    img = nib.load(filepath) #reads file
    return img.get_fdata().astype(np.float32) #obtain and return a floating point array with (float32) for consistency 



###=====================Preprocessing utilities =======================
def clip_intensity(img, lower=0.5, upper=99.5):
    '''
    Clips image intensities to the specified percentiles (lower=0.5, upper=99.5)
    '''
    low, high = np.percentile(img, [lower, upper]) # Computes percentile values
    return np.clip(img, low, high) # clips to remove extreme outliers and return results

def normalize(img, mask=None):
    '''
    Zero-mean, unit-variance normalization within a brain mask region
    '''
    #If no mask provided, defines mask as img > 0
    if mask is None:
        mask = img > 0
    if np.sum(mask) == 0:
        return img
    # Compute mean and std over with mask applied --> img[mask].
    mean = np.mean(img[mask])
    std  = np.std(img[mask])
    # If std is too small, subtracts only the mean.Otherwise scales (img - mean) / std
    if std < 1e-6:
        return img - mean
    normalized = (img - mean) / std
    return np.clip(normalized, -5, 5) #Clips final values to [-5, 5] to avoid extreme intensities


def resample_to_shape(img, target_shape=(128, 128, 128), mode="trilinear"):
    '''
    Resamples 3D or 4D volumes to a common shape using PyTorch’s interpolate
    Supports "trilinear" or "nearest" modes; uses align_corners=False for trilinear.

    '''
    if img.ndim == 3: # For 3D (D,H,W)
        img_t = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()  #Adds batch & channel dimensions: (1,1,D,H,W)
        if mode == "nearest":
            out = F.interpolate(img_t, size=target_shape, mode=mode) #Calls F.interpolate with size=target_shape
        else:
            out = F.interpolate(img_t, size=target_shape, mode=mode, align_corners=False) #Trilinear
        return out.squeeze().numpy()
    elif img.ndim == 4:  # For 4D (C,D,H,W)
        resampled = [resample_to_shape(img[c], target_shape, mode) for c in range(img.shape[0])] #Resamples each channel independently 
        return np.stack(resampled, axis=0)                                                       #and stacks back.
    else:
        raise ValueError(f"Unsupported img shape {img.shape}, expected 3D or 4D.")

def crop_to_mask(img, mask, margin=10):
    '''
    -->input: image and corresponding mask 
    Crops volume to the tightest bounding box around the mask plus a margin.
    <--Returns cropped (img, mask).
    '''
    coords = np.array(np.nonzero(mask)) #Finds all nonzero mask coordinates
    if coords.size == 0:
        return img, mask
    #Computes min_coords and max_coords per axis, expanded by margin.
    min_coords = np.maximum(coords.min(axis=1) - margin, 0)
    max_coords = np.minimum(coords.max(axis=1) + margin + 1, np.array(img.shape))
    #Constructs slice objects for each dimension
    slices = [slice(min_coords[i], max_coords[i]) for i in range(3)]
    return img[slices[0], slices[1], slices[2]], mask[slices[0], slices[1], slices[2]] #Returns cropped (img, mask).

def one_hot_encode(mask, num_classes=4):
    '''
    Converts integer labels to a one-hot encoded array.
    Original Brats Label: background (0), Non-Enh (1), Edema(2), ET(4) 
    After one hot: background (0), Non-Enh (1), Edema(2), ET(3) 
    '''
    mask = mask.astype(np.int32)
    mask[mask == 4] = 3 #Maps any label 4 to 3 (to unify ET label).
    return np.eye(num_classes)[mask]  # (D,H,W,C)

# ===== Cache Builder =========================================
def build_cache(image_paths, mask_paths, 
                target_shape=(128,128,128), 
                patch_size=(96,96,96), 
                num_patches=8,
                out_dir="cache"):
    os.makedirs(os.path.join(out_dir, "volumes"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "patches"), exist_ok=True)

    for i in tqdm(range(len(image_paths)), desc="Caching patients"):
        pid = os.path.basename(os.path.dirname(image_paths[i][0]))

        # --- Load + clip (no normalize yet)
        modalities = []
        for path in image_paths[i]:
            img = load_nifti(path)            
            img = clip_intensity(img)
            modalities.append(img)
        img = np.stack(modalities, axis=0)     

        if img.shape[1] != img.shape[2]:  
            img = np.transpose(img, (0, 3, 1, 2))
        mask = load_nifti(mask_paths[i])
        if mask.shape[0] != img.shape[1]:  # (H,W,D) -> (D,H,W)
            mask = np.transpose(mask, (2, 0, 1))

        # --- Crop once using mask; apply same slices to all channels
        brain_mask = img[0] > 0
        coords = np.array(np.nonzero(brain_mask))
        if coords.size == 0:
            slices = [slice(0, s) for s in brain_mask.shape]
        else:
            margin = 10
            minc = np.maximum(coords.min(axis=1) - margin, 0)
            maxc = np.minimum(coords.max(axis=1) + margin + 1, np.array(brain_mask.shape))
            slices = [slice(minc[i], maxc[i]) for i in range(3)]
        img  = img[:, slices[0], slices[1], slices[2]]
        mask = mask[slices[0], slices[1], slices[2]]

        # --- Resample (images: trilinear, mask: nearest)
        img  = resample_to_shape(img,  target_shape, mode="trilinear")
        mask = resample_to_shape(mask, target_shape, mode="nearest")

        # --- Normalize AFTER resample, using FLAIR foreground
        brain_mask = img[0] > 0
        for c in range(img.shape[0]):
            img[c] = normalize(img[c], mask=brain_mask)

        # --- Save P2 (full volume)
        np.savez_compressed(os.path.join(out_dir, "volumes", f"{pid}.npz"),
                            vol=img.astype(np.float32),
                            seg=mask.astype(np.uint8))

        # --- Save P3 (random patches)
        patch_dir = os.path.join(out_dir, "patches", pid)
        os.makedirs(patch_dir, exist_ok=True)
        d, h, w = patch_size
        _, D, H, W = img.shape
        for j in range(num_patches):
            z = np.random.randint(0, D - d + 1) if D > d else 0
            y = np.random.randint(0, H - h + 1) if H > h else 0
            x = np.random.randint(0, W - w + 1) if W > w else 0
            img_patch  = img[:, z:z+d, y:y+h, x:x+w]
            mask_patch = mask[z:z+d, y:y+h, x:x+w]
            mask_1h    = one_hot_encode(mask_patch, num_classes=4)
            np.savez_compressed(os.path.join(patch_dir, f"{pid}_patch{j}.npz"),
                                vol=img_patch.astype(np.float32),
                                seg=mask_1h.astype(np.uint8))

    print(f"Cache built at: {out_dir}/volumes (P2) and {out_dir}/patches (P3)")


# Shared Augmentation Function
def apply_augmentations(img_patch, mask_patch):
    # flips
    if random.random() > 0.5:
        img_patch, mask_patch = np.flip(img_patch, 1).copy(), np.flip(mask_patch, 0).copy()
    if random.random() > 0.5:
        img_patch, mask_patch = np.flip(img_patch, 2).copy(), np.flip(mask_patch, 1).copy()
    if random.random() > 0.5:
        img_patch, mask_patch = np.flip(img_patch, 3).copy(), np.flip(mask_patch, 2).copy()
    # rotations in-plane
    if random.random() > 0.5:
        k = random.choice([1,2,3])
        img_patch  = np.rot90(img_patch,  k, axes=(2,3)).copy()
        mask_patch = np.rot90(mask_patch, k, axes=(0,1)).copy()
    # intensity jitter
    if random.random() > 0.5:
        scale = random.uniform(0.9, 1.1)
        shift = random.uniform(-0.1, 0.1)
        img_patch = img_patch * scale + shift
    # gaussian noise
    if random.random() > 0.5:
        noise = np.random.normal(0, 0.01, img_patch.shape)
        img_patch = img_patch + noise
    return img_patch, mask_patch

# Dataset Classes - Pipeline 1
class BraTSDatasetP1(Dataset):  # P1 : on the fly
    def __init__(self, image_paths, mask_paths, 
                 target_shape=(128,128,128), 
                 patch_size=(96,96,96), 
                 augment=True, 
                 num_classes=4):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.target_shape = target_shape
        self.patch_size   = patch_size
        self.augment      = augment
        self.num_classes  = num_classes

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # --- Load + clip (no normalize yet)
        modalities = []
        for path in self.image_paths[idx]:
            arr = load_nifti(path)          # (H,W,D)
            arr = clip_intensity(arr)
            modalities.append(arr)
        img = np.stack(modalities, axis=0)  # (4,H,W,D) -> (4,D,H,W)
        img = np.transpose(img, (0,3,1,2))

        mask = load_nifti(self.mask_paths[idx])  # (H,W,D) -> (D,H,W)
        mask = np.transpose(mask, (2,0,1))

        # --- Crop once using mask; apply to all channels
        brain_mask = (img[0] > 0).astype(np.uint8)
        coords = np.array(np.nonzero(brain_mask))
        if coords.size == 0:
            slices = [slice(0, s) for s in brain_mask.shape]
        else:
            margin = 20
            minc = np.maximum(coords.min(axis=1) - margin, 0)
            maxc = np.minimum(coords.max(axis=1) + margin + 1, np.array(brain_mask.shape))
            slices = [slice(minc[i], maxc[i]) for i in range(3)]
        img  = img[:, slices[0], slices[1], slices[2]]
        mask = mask[slices[0], slices[1], slices[2]]

        # --- Resample
        img  = resample_to_shape(img,  self.target_shape, mode="trilinear")
        mask = resample_to_shape(mask, self.target_shape, mode="nearest")

        # --- Normalize (use FLAIR foreground)
        brain_mask = img[0] > 0
        for c in range(img.shape[0]):
            img[c] = normalize(img[c], mask=brain_mask)

        # --- Patch (center for debugging; switch to random for training)
        d,h,w = self.patch_size
        _, D,H,W = img.shape
        pos = np.argwhere(mask > 0)
        use_fg = (len(pos) > 0) and (random.random() < 0.5)   # 50% 前景patch
        if use_fg:
            cz, cy, cx = pos[random.randrange(len(pos))]
            z = max(0, min(cz - d//2, D - d))
            y = max(0, min(cy - h//2, H - h))
            x = max(0, min(cx - w//2, W - w))
        else:
            z = random.randint(0, max(D - d, 0))
            y = random.randint(0, max(H - h, 0))
            x = random.randint(0, max(W - w, 0))

        img_patch  = img[:, z:z+d, y:y+h, x:x+w]
        mask_patch = mask[z:z+d, y:y+h, x:x+w]

        # --- One-hot
        mask_1h = one_hot_encode(mask_patch, self.num_classes)  # (d,h,w,C)

        # --- Augment (if enabled)
        if self.augment:
            img_patch, mask_1h = apply_augmentations(img_patch, mask_1h)

        # --- Tensors
        img_t  = torch.tensor(img_patch).float()                   # (4,d,h,w)
        mask_t = torch.tensor(mask_1h).permute(3,0,1,2).float()    # (C,d,h,w)

        return img_t, mask_t
        
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





