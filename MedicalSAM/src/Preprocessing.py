import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


# Dataset設置
class KvasirDataset(Dataset):
    def __init__(self, images_data, masks_data, img_size=256, mode="train"):
        self.images_data = images_data
        self.masks_data = masks_data
        self.mode = mode
        self.img_size = img_size

        # 建立Augmentation方式(記得只用於Training)
        self.transforms = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5
                ),
                A.RandomBrightnessContrast(p=0.2),
                # 模擬組織變形
                A.ElasticTransform(
                    alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5
                ),
            ]
        )

        # 建立標準化方式(包含val)
        # 設定為ImageNet的標準化
        self.normalize = A.Compose(
            [
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, idx):
        # 抓取每張圖片和Mask路徑
        img_path = self.images_data[idx]
        mask_path = self.masks_data[idx]

        # convert('RGB') 確保是 3 channel, convert('L') 確保 Mask 是灰階
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Resizing + 插值
        # 原圖使用 BICUBIC 平滑過渡
        # Mask使用 NEAREST，並且二值化
        image = image.resize((self.img_size, self.img_size), resample=Image.BICUBIC)
        mask = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)

        # 建立Augmentation(只用於Training)，記得Augmentation只吃Numpy array
        image = np.array(image)
        mask = np.array(mask)
        mask = np.where(mask > 127, 1, 0).astype(np.uint8)

        if self.mode == "train":
            augmented = self.transforms(image=np.array(image), mask=np.array(mask))
            image = augmented["image"]
            mask = augmented["mask"]

        # MedSAM的BBox轉換(Prompt)，避免Transforms完畢後BBox和圖片對不起來
        # 不用使用BBox，直接使用mask算出BBox
        row_idx, col_idx = np.where(mask == 1)
        if len(row_idx) != 0:
            row_min, row_max = np.min(row_idx), np.max(row_idx)
            col_min, col_max = np.min(col_idx), np.max(col_idx)
            # 加入隨機擾動
            Height, Width = mask.shape
            row_min = max(0, row_min - np.random.randint(0, 10))
            col_min = max(0, col_min - np.random.randint(0, 10))
            row_max = min(Height, row_max + np.random.randint(0, 10))
            col_max = min(Width, col_max + np.random.randint(0, 10))
            # [x1, y1, x2, y2]
            bbox = np.array([col_min, row_min, col_max, row_max])
        else:
            bbox = torch.from_numpy(
                np.array([0, 0, self.img_size, self.img_size])
            ).float()

        # 建立標準化和toTensor，Mask不要亂normalize，需要是0,1的精確值
        image = self.normalize(image=image)["image"]
        # 把 [H, W] 變成 [1, H, W]
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        bbox = torch.from_numpy(bbox).float()

        return image, mask, bbox
