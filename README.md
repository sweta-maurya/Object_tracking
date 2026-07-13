# Object_tracking

# 🚀 Object Detection and Object Tracking using YOLO

This project demonstrates real-time **Object Detection** and **Object Tracking** using the YOLO (You Only Look Once) model with OpenCV. It can detect multiple objects in images or videos and track them across frames with unique IDs.

## 📌 Features

- Real-time object detection
- Multi-object tracking
- Unique ID assignment for tracked objects
- Bounding boxes with class labels
- Confidence score display
- Video and webcam support
- Fast and efficient performance using YOLO

## 🛠️ Technologies Used

- Python
- OpenCV
- YOLO (You Only Look Once)
- NumPy

## 📂 Project Structure

```
Object-Detection-and-Tracking/
│
├── object_detection.py      # Detects objects in images/videos
├── object_tracking.py       # Tracks detected objects across video frames
├── requirements.txt         # Project dependencies (optional)
├── yolov4.weights           # YOLO weights (not uploaded if too large)
├── yolov4.cfg               # YOLO configuration file
├── coco.names              # Class labels
└── README.md
```

## 📖 File Description

### 1. object_detection.py

This file performs object detection using the YOLO model.

**Functions:**
- Loads the YOLO model
- Detects objects in each frame
- Draws bounding boxes
- Displays object labels and confidence scores

---

### 2. object_tracking.py

This file performs object tracking after detection.

**Functions:**
- Detects objects using YOLO
- Tracks objects across consecutive frames
- Assigns unique IDs to each object
- Maintains object identity while moving

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Object-Detection-and-Tracking.git
```

Move into the project directory:

```bash
cd Object-Detection-and-Tracking
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Running the Project

### Object Detection

```bash
python object_detection.py
```

### Object Tracking

```bash
python object_tracking.py
```

## 📊 Output

The application displays:

- Detected objects
- Bounding boxes
- Object labels
- Confidence scores
- Tracking IDs (for tracking)

Example:

```
Person | ID: 3 | Confidence: 0.94
Car    | ID: 1 | Confidence: 0.98
```

## 📷 Sample Use Cases

- Smart Surveillance
- Traffic Monitoring
- Vehicle Tracking
- Crowd Analysis
- Security Systems
- Autonomous Vehicles

## 📈 Future Improvements

- Improve tracking accuracy
- Support custom-trained YOLO models
- Export tracking data to CSV
- Real-time analytics dashboard
- DeepSORT integration
- GPU acceleration

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Push the branch
5. Open a Pull Request

## 📄 License

This project is intended for educational and learning purposes.
