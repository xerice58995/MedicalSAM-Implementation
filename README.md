MedSAM Polyp Segmentation

This project implements an automated endoscopic polyp segmentation system based on MedSAM (Segment Anything Model for Medical Images). By freezing the pre-trained Image Encoder and fine-tuning the Mask Decoder, combined with Bounding Box prompt learning and LCC (Largest Connected Component) post-processing, the system achieves high-precision segmentation on the Kvasir-SEG dataset.

📂 Project Structure

MedicalSAM/
├── Weights/               # Pre-trained weights (e.g., sam_vit_b_01ec64.pth)
├── logs/                  # TensorBoard logs
├── Training.py            # Main script: Training, Validation, and Testing loops
├── Model.py               # Model architecture: MedSAM freeze & fine-tune logic
├── Datasplit.py           # Data processing: Train/Val/Test split (8:1:1)
├── Preprocessing.py       # Data loading: Dataset Class & Albumentations
├── LossFunction.py        # Loss function: BCE + Dice Loss combination
├── EvaluationMetrics.py   # Metrics: Dice, IoU, HD95, LCC post-processing
├── requirements.txt       # Dependencies
└── README.md              # Project documentation


🚀 Quick Start

1. Installation

Ensure Python 3.8+ and PyTorch are installed. Install the required packages:

pip install torch torchvision
pip install segment-anything
pip install opencv-python scipy matplotlib tqdm tensorboard albumentations scikit-learn


2. Prepare Weights

This project uses SAM (ViT-B) as the base model. Download the weights and place them in the Weights/ directory:

Download Link (sam_vit_b_01ec64.pth)

3. Prepare Dataset

Download the Kvasir-SEG dataset and ensure the directory structure matches below (or update root_dir in Training.py):

MedicalSAM/
└── Kvasir-SEG/
    └── data/
        ├── images/
        └── masks/


4. Training

Run the main script to start training, validation, and testing:

python Training.py

