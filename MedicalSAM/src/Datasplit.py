import glob
import os
import random

from sklearn.model_selection import train_test_split


def get_split(root_dir, seed=42):
    image_dir = os.path.join(root_dir, "data/Kvasir-SEG/images")
    mask_dir = os.path.join(root_dir, "data/Kvasir-SEG/masks")

    # 用glob取得所有圖片路徑
    images = glob.glob(os.path.join(image_dir, "*"))
    masks = glob.glob(os.path.join(mask_dir, "*"))

    images.sort()
    masks.sort()

    if len(images) != len(masks):
        raise ValueError("Number of images and masks must be the same.")

    train_imgs, temp_imgs = train_test_split(images, test_size=0.2, random_state=seed)
    train_masks, temp_masks = train_test_split(masks, test_size=0.2, random_state=seed)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=seed)
    val_masks, test_masks = train_test_split(
        temp_masks, test_size=0.5, random_state=seed
    )

    return train_imgs, train_masks, val_imgs, val_masks, test_imgs, test_masks
