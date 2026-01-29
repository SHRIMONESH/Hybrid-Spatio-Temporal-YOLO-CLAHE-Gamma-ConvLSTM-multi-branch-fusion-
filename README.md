# Nighttime Vehicle Detection using YOLOv8 🚗🌑

![YOLOv8](https://img.shields.io/badge/YOLO-v8-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Python](https://img.shields.io/badge/Python-3.10%2B-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📌 Project Overview

**Nighttime Vehicle Detection** is a computer vision project aimed at enhancing road safety and surveillance systems by accurately detecting vehicles in low-light and nighttime conditions. Traditional object detectors often struggle with the glare of headlights, street lamps, and low contrast. This project overcomes those challenges by fine-tuning **Ultralytics YOLOv8 (Nano and Small)** models on a specialized dataset.

The repository features a complete pipeline:
1.  **Custom Data Parsing:** A specialized script to convert raw `ground_truth.txt` annotations into the standard YOLO format.
2.  **Model Training:** Configurations for training `yolov8n` and `yolov8s` models.
3.  **Performance Evaluation:** Automated generation of Confusion Matrices, PR Curves, and F1-Score plots.

## 🏗️ Technical Architecture & Tech Stack

* **Core Framework:** [Ultralytics YOLOv8](https://docs.ultralytics.com/)
* **Deep Learning Backend:** [PyTorch](https://pytorch.org/)
* **Image Processing:** OpenCV (`cv2`), Pillow (`PIL`)
* **Data Manipulation:** NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Google Colab (optimized for Tesla T4 GPU)

## 📂 Project Structure

The project relies on a specific directory structure to handle the custom dataset and training results via Google Drive.

```text
├── NITTAPP3.ipynb                  # Main execution notebook
├── nighttime_vehicle_dataset/      # Dataset Source
│   ├── images/                     # Raw nighttime images (img_xxxxx.jpg)
│   └── ground_truth.txt            # Annotations (Filename Num_Vehicles X1 Y1 W H...)
├── nighttime_vehicle_detection/    # Local working directory
│   └── labels/                     # Generated YOLO formatted labels (.txt)
├── nighttime_vehicle_results/      # Output Directory
│   ├── yolov8n_nighttime/          # Nano model weights & logs
│   ├── yolov8s_nighttime/          # Small model weights & logs
│   └── runs/                       # Tensorboard events
└── README.md
