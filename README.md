# Real-Time Object Detection using YOLOv3-Tiny and OpenCV

This project implements real-time object detection using a webcam with the YOLOv3-Tiny deep learning model and OpenCV’s DNN module.  
It is optimized to run efficiently on CPU-only systems, making it ideal for academic projects and low-end machines.

---

## Project Overview

The system captures live video from a webcam, processes each frame using the YOLOv3-Tiny object detection model, and identifies multiple objects in real time. Detected objects are displayed with bounding boxes, class labels, and confidence scores.

YOLOv3-Tiny is a lightweight version of YOLOv3, designed for faster inference with minimal performance loss.

---

## Features

- Real-time object detection using webcam
- Fast detection using YOLOv3-Tiny
- Detects 80+ common objects from the COCO dataset
- Displays bounding boxes with labels and confidence scores
- Runs entirely on CPU (no GPU required)
- Optimized for better performance on low-end systems
- Clean and readable Python code
- Suitable for mini-projects and major academic projects

---

## Technologies Used

- Python 3
- OpenCV (DNN module)
- NumPy
- YOLOv3-Tiny
- COCO Dataset (class labels)

---

## Project Structure

- Object-Detection/
  - Object_Detection.py  
    Main Python script for real-time object detection

  - yolov3-tiny.cfg  
    YOLOv3-Tiny configuration file

  - yolov3-tiny.weights  
    Pre-trained YOLOv3-Tiny weights file

  - coco.names  
    COCO dataset class labels

  - requirements.txt  
    Python dependencies

  - README.md  
    Project documentation


---

## How It Works

1. Captures live video feed from the webcam.
2. Converts each frame into a blob for neural network processing.
3. Passes the blob through the YOLOv3-Tiny network.
4. Extracts detected objects with confidence above a threshold.
5. Applies Non-Maximum Suppression (NMS) to remove overlapping boxes.
6. Displays the final detections on the video stream.

---

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/Object-Detection-YOLO.git
cd Object-Detection-YOLO
Step 2: Install Dependencies
bash
Copy code
pip install -r requirements.txt
Running the Project
Make sure all files are present in the same directory.

bash
Copy code
python Object_Detection.py
Press ESC to exit the application.
