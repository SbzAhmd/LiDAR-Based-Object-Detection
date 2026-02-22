# 🚗 LiDAR-Based Object Detection

A LiDAR-based object detection system designed to detect and analyze objects from LiDAR point cloud data. This project processes raw LiDAR inputs, performs inference using a trained model, and visualizes detections for traffic and object analysis.

---

## 📌 Overview

This project focuses on:

- 📡 Loading and preprocessing LiDAR point cloud data  
- 🧠 Performing object detection using a trained deep learning model  
- 📊 Visualizing detected objects in 2D/3D space  
- 🚦 Analyzing traffic-related metrics  

The system is modular and structured for easy experimentation and extension

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
