import os

import numpy as np
import torch
import torch.optim as optim
import torchvision.utils as vutils
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Datasplit import get_split
from EvaluationMetrics import MedicalMetrics, lcc_transform
from LossFunction import LossFunction, dice_score
from Model import MedSAM_Model
from Preprocessing import KvasirDataset

# 1. ----Dataloader ----

root_dir = "MedicalSAM"

splits = get_split(root_dir, seed=42)
train_imgs, train_masks, val_imgs, val_masks, test_imgs, test_masks = splits

train_dataset = KvasirDataset(train_imgs, train_masks, img_size=256, mode="train")
val_dataset = KvasirDataset(val_imgs, val_masks, img_size=256, mode="val")
test_dataset = KvasirDataset(test_imgs, test_masks, img_size=256, mode="test")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print("Datasets and DataLoaders created successfully!")

# 2. ----Hyperparameters ----
parameters = {
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "batch_size": 16,
    "weight_decay": 1e-3,
    "patience": 10,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 3. ----Model ----
model = MedSAM_Model(freeze_encoder=True).to(device)
metric_tool = MedicalMetrics()
loss_criterion = MedSAM_Loss().to(device)

# MedSAM 凍結了 Encoder，不需要把model.parameters() 全部丟給 optimizer，使用filter濾掉不需要更新grad的參數
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=parameters["learning_rate"],
    weight_decay=parameters["weight_decay"],
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=parameters["num_epochs"], eta_min=1e-6
)

# Mixed Precision Scaler (AMP)，能讓效能增加且減少記憶體使用
# scaler在反向傳播時使用
scaler = torch.cuda.amp.GradScaler()
writer = SummaryWriter(log_dir="logs/experiment_1")

# 4. ----Training Loop ----
best_dice = 0.0
patience_counter = 0
global_step = 0

print(f"-------於{device}上開始訓練模型-------")

for epoch in range(parameters["num_epochs"]):
    print(f"第 {epoch + 1} / {parameters['num_epochs']} 個 Epoch 開始訓練")

    # ----Training----
    model.train()
    train_loss = 0.0
    loop = tqdm(
        train_loader, desc=f"{epoch + 1} / {parameters['num_epochs']}[Training]"
    )

    for images, masks, bbox in loop:
        optimizer.zero_grad()

        images = images.to(device)
        masks = masks.to(device)
        bbox = bbox.to(device)

        # Forward (使用 AutoCast 開啟混合精度)
        with autocast():
            preds = model(images, bbox)
            loss = loss_criterion(preds, masks)

        # Backward(使用Scaler節省記憶體)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        writer.add_scalar("Train_Loss / Steps", loss.item(), global_step=global_step)
        loop.set_postfix(loss=f"{loss.item():.4f}")

        global_step += 1

    # 計算平均loss
    avg_train_loss = train_loss / len(train_loader)
    writer.add_scalar("Train_Loss / Epoch", avg_train_loss, global_step=epoch)

    # ----Validation----
    model.eval()
    val_loss = 0.0

    # 本次實驗的Dice和HD95分數，累積所有 batch 的分數，最後取平均
    val_metrics = {"dice": [], "hd95": [], "lcc_dice": [], "lcc_hd95": []}

    # 抓圖片對照出來做視覺化
    visual_image, visual_mask, visual_pred = None, None, None

    with torch.no_grad():
        for images, masks, bbox in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)
            bbox = bbox.to(device)

            preds = model(images, bbox)

            # ----計算Loss----
            # 記得我們要使用之前定義的dice score計算函式來計算Loss，Dice Loss應該越大越好
            # 要記得這邊的dice score是BCE + Dice給loss計算用的，不是Metric的結算Dice score
            batch_dice = dice_score(preds, masks)
            val_loss += batch_dice

            # ----存圖做視覺化----
            if visual_image is None:
                visual_image = images[0].detach().cpu()
                visual_mask = masks[0].detach().cpu()
                visual_pred = preds[0].detach().cpu()

            # ----計算 Evaluation Metrics----
            # 1. pred轉成0 / 1 mask
            pred_to_mask = torch.sigmoid(preds)
            pred_to_mask = (pred_to_mask > 0.5).float()

            # 2. 搬到CPU轉成Numpy給Metrics吃
            # shape: [B, 1, H, W] -> [B, H, W] (squeeze 掉 channel)
            np_pred_to_mask = pred_to_mask.cpu().numpy().squeeze(1)
            np_target = masks.cpu().numpy().squeeze(1)

            # 逐張計算Metrics
            for i in range(len(np_pred_to_mask)):
                p_img = np_pred_to_mask[i]
                t_img = np_target[i]

                p_img_clean = lcc_transform(p_img)  # LCC濾掉雜訊的圖

                dice_metric = metric_tool.calculate_dice(p_img, t_img)
                hd95_metric = metric_tool.calculate_hd95(p_img, t_img)
                lcc_dice_metric = metric_tool.calculate_dice(p_img_clean, t_img)
                lcc_hd95_metric = metric_tool.calculate_hd95(p_img_clean, t_img)

                val_metrics["dice"].append(dice_metric)
                val_metrics["hd95"].append(hd95_metric)
                val_metrics["lcc_dice"].append(lcc_dice_metric)
                val_metrics["lcc_hd95"].append(lcc_hd95_metric)

        # ----結算平均Metric Score----
        avg_dice_metric = np.mean(val_metrics["dice"])
        avg_hd95_metric = np.mean(val_metrics["hd95"])
        avg_lcc_dice_metric = np.mean(val_metrics["lcc_dice"])
        avg_lcc_hd95_metric = np.mean(val_metrics["lcc_hd95"])

        writer.add_scalar("Dice / Epoch", avg_val_dice, global_step=epoch)
        writer.add_scalar("Avg Dice / Epoch", avg_dice_metric, global_step=epoch)
        writer.add_scalar("Avg HD95 / Epoch", avg_hd95_metric, global_step=epoch)
        writer.add_scalar(
            "LCC Avg Dice / Epoch", avg_lcc_dice_metric, global_step=epoch
        )
        writer.add_scalar(
            "LCC Avg HD95 / Epoch", avg_lcc_hd95_metric, global_step=epoch
        )

        scheduler.step()

        # 建構視覺化對比
        if visual_image is not None:
            img = visual_image
            label = visual_mask
            pred = (torch.sigmoid(visual_pred) > 0.5).float()

            # 把mask和label轉成RGB三通道後再並排視覺化
            comparison_img = vutils.make_grid(
                [img, label.repeat(3, 1, 1), pred.repeat(3, 1, 1)],
                nrow=3,
                normalize=True,
            )
            writer.add_image("Comparison / Epoch", comparison_img, global_step=epoch)

        # ----End of Epoch Processing----
        print(
            f"Epoch {epoch + 1} finished | Train Loss: {avg_train_loss:.4f} |Val Loss: {avg_val_dice:.4f} | Val Dice: {avg_dice_metric:.4f} |Val HD95: {avg_hd95_metric:.4f} "
        )

        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(model.state_dict(), "../Weights/best_model.pth")
            print(f"New best dice score: {avg_val_dice:.4f}. Model Saved")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= parameters["patience"]:
            print("No progress. Early stopping!")
            break

writer.close()
print("Training completed.")

# ----Test Loop----
print(" Starting Testing Phase...")
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

test_metrics = {"dice": [], "hd95": [], "lcc_dice": [], "lcc_hd95": []}

with torch.no_grad():
    for images, masks, bbox in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        masks = masks.to(device)
        bbox = bbox.to(device)

        preds = model(images, bbox)
        preds = (torch.sigmoid(preds) > 0.5).floatß()

        # 一樣搬回CPU來計算Metrics
        np_preds = preds.cpu().numpy().squeeze(1)
        np_masks = masks.cpu().numpy().squeeze(1)

        for i in range(len(np_preds)):
            p_img = np_preds[i]
            t_img = np_masks[i]
            p_img_clean = lcc_transform(p_img)

            test_metrics["dice"].append(metrics.calculate_dice(p_img, t_img))
            test_metrics["hd95"].append(metrics.calculate_hd95(p_img, t_img))
            test_metrics["lcc_dice"].append(metrics.calculate_dice(p_img_clean, t_img))
            test_metrics["lcc_hd95"].append(metrics.calculate_hd95(p_img_clean, t_img))

final_dice_mean = np.mean(test_metrics["dice"])
final_dice_std = np.std(test_metrics["dice"])
final_hd95_mean = np.mean(test_metrics["hd95"])
final_lcc_dice_mean = np.mean(test_metrics["lcc_dice"])
final_lcc_dice_std = np.std(test_metrics["lcc_dice"])
final_lcc_hd95_mean = np.mean(test_metrics["lcc_hd95"])


print(f"Final Dice Mean: {final_dice_mean:.4f} | Final Dice Std: {final_dice_std:.4f}")
print(
    f"Final Dice Mean(LCC): {final_lcc_dice_mean:.4f} | Final Dice Std(LCC): {final_lcc_dice_std:.4f}"
)
print(f"Final HD95 Mean: {final_hd95_mean:.4f}")
print(f"Final HD95 Mean(LCC): {final_lcc_hd95_mean:.4f}")
