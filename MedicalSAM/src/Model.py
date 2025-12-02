# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import torchvision
from segment_anything import sam_model_registry
from torch import nn


# Model吃圖片和BBox，輸出分割Mask
class MedSAM_Model(nn.Module):
    """
    Args:
        image: [B, 3, 1024, 1024] - 輸入圖片
        bbox:  [B, 4] - BBox 座標 [x1, y1, x2, y2]
    Returns:
        masks: [B, 1, 1024, 1024] - 預測的分割圖
    """

    def __init__(self, freeze_encoder=True):
        super(MedSAM_Model, self).__init__()
        # Meta MedSAM Pre-trained Model(ViT-B) for Kvasir-SEG
        self.sam = sam_model_registry["vit_b"](
            checkpoint="../Weights/sam_vit_b_01ec64.pth"
        )

        # 將 Image Encoder 和 Prompt Encoder 凍結，只訓練 Mask Decoder
        if freeze_encoder:
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False
            for param in self.sam.prompt_encoder.parameters():
                param.requires_grad = False

            # Mask Decoder 梯度計算保持開啟

    def forward(self, image, bbox):
        # 1. ------Image Encoder------
        # Image Encoder 部分不計算梯度，節省記憶體
        with torch.no_grad():
            image_embeddings = self.sam.image_encoder(image)

        # 2. ------BBox(Prompt) Encoder------
        # sparse稀疏特徵(box)，dense密集特徵(像素mask)，在此我們只用box
        # SAM 需要 BBox 形狀為 [Batch, box數量, 座標]，
        # 例如 [16, 1, 4]-->「共16張圖，一個框框，框框座標為4點[x1,y1,x2,y2]」
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=bbox.unsqueeze(1),  # [B, 4] -> [B, 1, 4]
            masks=None,
        )

        # 3. ------Mask Decoder------
        # 使用 Mask Decoder 預測分割結果，需要計算梯度
        # 由於image_embeddings沒有位置資訊，需要搭配image_pe作為位置資訊的基板，
        # 可以讓Decoder畫出Masks時知道位置資訊

        # 取得位置編碼 (Shape: [1, 256, 64, 64])
        dense_pe = self.sam.prompt_encoder.get_dense_pe()

        # 將位置編碼擴展 (Repeat) 到與目前的 Batch Size (image_embeddings.shape[0]) 一致
        # Shape 變更: [1, 256, 64, 64] -> [B, 256, 64, 64]
        dense_pe = dense_pe.repeat(image_embeddings.shape[0], 1, 1, 1)

        prediction_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=dense_pe,  # 使用擴展後的 dense_pe
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # 4. ------Resize to Original Size------
        # 將prediction_masks解析度升回來

        orginal_size = image.shape[-2:]  # (image: [B, channels, H, W])
        # ***interpolate用來調整Tensor大小***，不能用Resize(調整PIL Image)取代
        # 用bilinear是bicubic都可以，進行平滑過渡，和preprocessing Mask使用NEAREST二值化不同，
        # resized_masks的Tensor需要平滑來讓梯度流動
        # 最終輸出時會加一個 Threshold 切回 0 和 1，
        resized_masks = F.interpolate(
            prediction_masks,
            size=orginal_size,
            mode="bilinear",
            align_corners=False,
        )

        return resized_masks
