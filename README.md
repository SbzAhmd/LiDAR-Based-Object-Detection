# 🚗 LiDAR-Based Object Detection

A LiDAR-based object detection system designed to detect and analyze objects from LiDAR point cloud data. This project processes raw LiDAR inputs, performs inference using a trained model, and visualizes detections for traffic and object analysis.

---

## 📌 Overview

This project focuses on:

- 📡 Loading and preprocessing LiDAR point cloud data  
- 🧠 Performing object detection using a trained deep learning model  
- 📊 Visualizing detected objects in 2D/3D space  
- 🚦 Analyzing traffic-related metrics  

The system is modular and structured for easy experimentation and extension.

---

## 🗂 Project Structure

LiDAR-Based-Object-Detection/
│
├── lidar_loader.py # Loads and preprocesses LiDAR data
├── model_inference.py # Handles model loading and inference
├── test_detection.py # Testing script for detection
├── traffic_analyzer.py # Traffic metrics and analysis
├── normal_visualization.py # Standard visualization utilities
├── visualizer.py # Detection visualization
├── main.py # Main execution script
├── requirements.txt # Project dependencies
├── info.txt # Additional project info
└── .gitignore



---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/SbzAhmd/LiDAR-Based-Object-Detection.git
cd LiDAR-Based-Object-Detection

python -m venv venv
source venv/bin/activate      # On Linux/Mac
venv\Scripts\activate         # On Windows

pip install -r requirements.txt

python main.py

python test_detection.py
