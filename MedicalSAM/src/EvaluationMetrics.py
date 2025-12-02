import cv2
import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff


class MedicalMetrics:
    def __init__(self):
        pass

    def calculate_dice(self, pred, target):
        """
        args:
        pred: (H, W) numpy array, 0 or 1
        target: (H, W) numpy array, 0 or 1
        """
        intersection = (pred * target).sum()
        union = (pred + target).sum()
        if union == 0:
            return 1.0
        return (2 * intersection + 1e-5) / (union + 1e-5)

    def calculate_hd95(self, pred, target):
        """
        95% Hausdorff Distance(這邊先使用100%)
        預測邊界與真實邊界的最大不吻合程度
        """
        if np.sum(pred) == 0 or np.sum(target) == 0:
            if np.sum(pred) == 0 and np.sum(target) == 0:
                return 0
            else:
                h, w = pred.shape
                max_distance = np.sqrt(h**2 + w**2)
                return max_distance

        # directed_hausdorff不吃圖片，只吃標記的座標，所以要先轉換
        pred_coordinates = np.argwhere(pred > 0)
        target_coordinates = np.argwhere(target > 0)

        pred_to_target = directed_hausdorff(pred_coordinates, target_coordinates)[0]
        target_to_pred = directed_hausdorff(target_coordinates, pred_coordinates)[0]
        return max(pred_to_target, target_to_pred)


def lcc_transform(pred_mask):
    """
    Local Connected Component Transform
    將有小雜訊的mask轉換成單一大連通區域的mask，減少雜訊的影響
    connectedComponentsWithStats能回傳關鍵參數
    # num_labels: 區域總數 (含背景)
    # labels: 標記好的圖，每個連通區域會標上各自的num_labels
    # stats: 每個區域的統計資訊 [x, y, width, height, area]
    # centroids: 每個區域的質心座標 [x, y]
    """
    pred_mask = pred_mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        pred_mask, connectivity=8
    )

    if num_labels == 1:
        return pred_mask

    largest_label = np.argmax(stats[1:, 4]) + 1

    new_mask = np.zeros_like(pred_mask)  # 1. 先拿一張全黑的畫布
    new_mask[labels == largest_label] = 1  # 2. 把最大連通區域的位置標註好

    return new_mask
