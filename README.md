# MedSAM Polyp Segmentation

This project implements an automated endoscopic polyp segmentation system based on **MedSAM (Segment Anything Model for Medical Images)**. By freezing the pre-trained Image Encoder and fine-tuning the Mask Decoder, combined with Bounding Box prompt learning and **LCC (Largest Connected Component)** post-processing, the system achieves high-precision segmentation on the Kvasir-SEG dataset.

##  Project Structure

```text
MedicalSAM/
├── data/                      
│   └── Kvasir-SEG/            # 原始影像與標註，需自行下載
│
├── src/                       
│   ├── logs/                  # TensorBoard 訓練日誌
│   ├── Datasplit.py           # Train/Val/Test 資料分割（8:1:1）
│   ├── EvaluationMetrics.py   # 評估指標：Dice、HD95、LCC
│   ├── GPU_release.py         # 釋放 GPU 記憶體的工具程式
│   ├── LossFunction.py        # BCE + Dice Loss
│   ├── Model.py               # MedSAM 主模型，包含 encoder/decoder 與 fine-tune 設計
│   ├── Preprocessing.py       # Dataset class 與 Albumentations 資料前處理
│   └── Training.py            # 主訓練流程：Train / Validation / Testing loop
│   └── requirements.txt       # 所需套件清單
│
└── Weights/
    └── sam_vit_b_01ec64.pth   # 預訓練的 SAM 模型權重，需自行下載
```

##  Quick Start

### 1. Installation

Ensure Python 3.8+ and PyTorch are installed. Install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Prepare Weights

This project uses SAM (ViT-B) as the base model. Download the weights and place them in the `Weights/` directory:

* [Download Link (sam_vit_b_01ec64.pth)](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

### 3. Prepare Dataset

Download the **Kvasir-SEG** dataset and ensure the directory structure matches below:

```text
MedicalSAM/
└── Kvasir-SEG/
    └── data/
        ├── images/
        └── masks/
```

### 4. Training

Make sure to run the main script under the `src/` directory:

```bash
python Training.py
```
