import os
import shutil
import random
from tqdm import tqdm

# Paths
original_dataset_dir = 'malaria_dataset_original'
output_base_dir = 'malaria_dataset'

# Ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Classes
classes = ['Parasitized', 'Uninfected']

# Create output dirs
for split in ['train', 'val', 'test']:
    for cls in classes:
        os.makedirs(os.path.join(output_base_dir, split, cls), exist_ok=True)

# Distribute images
for cls in classes:
    src_dir = os.path.join(original_dataset_dir, cls)
    all_images = os.listdir(src_dir)
    random.shuffle(all_images)

    total = len(all_images)
    train_cut = int(total * train_ratio)
    val_cut = int(total * (train_ratio + val_ratio))

    splits = {
        'train': all_images[:train_cut],
        'val': all_images[train_cut:val_cut],
        'test': all_images[val_cut:]
    }

    for split, filenames in splits.items():
        for fname in tqdm(filenames, desc=f"{cls} → {split}"):
            src = os.path.join(src_dir, fname)
            dst = os.path.join(output_base_dir, split, cls, fname)
            shutil.copy2(src, dst)
