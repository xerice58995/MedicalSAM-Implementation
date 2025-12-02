import torch
import torch.nn as nn


def dice_score(pred, target, threshold=0.5):
    """
    計算驗證集上的 Dice Score (越高越好)
    preds: Logits [B, 1, H, W]
    targets: Labels [B, 1, H, W]
    """
    preds = torch.sigmoid(pred)
    preds = (preds > threshold).float()  # 二值化

    intersection = (preds * target).sum()
    union = preds.sum() + target.sum()
    dice = (2 * intersection + 1e-5) / (union + 1e-5)
    return dice.item()


# 使用BCE + Dice Loss作為損失函數，可補上Focal Loss增加困難樣本的權重
class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    # BCE Loss (Binary Cross Entropy) 每一個像素都當作獨立是非題
    # 會有像是Accuracy一樣不平衡問題，都猜0也會有高分
    def BCE_Loss(self, pred, target):
        return self.bce_loss(pred, target)

    # Dice Loss 衡量預測和真實標籤的重疊程度，精確但是梯度容易不穩
    """
    pred: (B, 1, H, W) - Logits (未經過 Sigmoid)
    target: (B, 1, H, W) - 0 或 1 的 Ground Truth
    """

    def dice_loss(self, pred, target, smooth=1e-5):
        pred = torch.sigmoid(pred)  # 轉成機率 0~1
        # Flatten: 拉平成一維向量，方便計算交集
        # -1 代表自動計算該維度元素數量，由於只給了一個元素，代表只有一維
        # 如果是.view(3, -1)，代表把Tensor變成3列，行數自動計算
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        # 分母是「A 的面積 + B 的面積」，由於交集算了兩次，所以分子是 2 * intersection以平衡
        dice = (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1 - dice

    def forward(self, pred, target):
        bce = self.BCE_Loss(pred, target)
        dice = self.dice_loss(pred, target)
        loss = 0.5 * bce + 0.5 * dice
        return loss
