# 📚 ClassEye — AI-Powered Educational Monitoring System

## 📑 Table of Contents

1. [Introduction](#introduction)
2. [Project Objectives](#project-objectives)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Project Structure](#project-structure)
6. [System Workflow](#system-workflow)
7. [API Routes](#api-routes)
8. [Image and Video Processing](#image-and-video-processing)
9. [Report Generation](#report-generation)
10. [Processing History](#processing-history)
11. [Project Benefits](#project-benefits)
12. [Required Investments](#required-investments)
13. [Final Considerations](#final-considerations)

---

## 📌 Introduction

**ClassEye** is a Flask-based system that uses **computer vision and artificial intelligence** to monitor student behavior in educational environments. It processes images and videos through YOLO (You Only Look Once) and facial recognition, generating automatic reports and visual data.

The system is designed to assist schools by promoting student focus, reducing distractions, and providing meaningful analytics for educators.

---

## 🎯 Project Objectives

* 🎥 **Classroom Monitoring:** Detects mobile phone usage, inattentive behavior, and classroom distractions.
* ⚙️ **Real-Time Analysis:** Automatically processes uploaded videos and images.
* 📊 **Engagement Reports:** Generates visual graphs to help teachers assess classroom attention.
* 🤖 **Face Recognition:** Identifies presence and focus of registered students.

---

## 🧰 Requirements

* Python 3.10+
* Flask
* OpenCV
* NumPy
* Matplotlib
* face\_recognition
* ultralytics (YOLO)

### Install Dependencies

```bash
pip install flask opencv-python-headless matplotlib numpy face-recognition ultralytics
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/brunoxzr/ClassEye.git
cd ClassEye

# (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# or
source venv/bin/activate     # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Then, access: [http://127.0.0.1:8080](http://127.0.0.1:8080)

---

## 🗂️ Project Structure

```
/ClassEye
├── app.py                # Main Flask application
├── detect.py             # Image/video processing functions
├── config.py             # Configuration settings
├── utils.py              # Detection and recognition helpers
├── uploads/              # Uploaded files
├── static/
│   ├── css/
│   ├── js/
│   ├── images/
│   └── reports/          # Generated graphs
├── templates/            # HTML pages
│   ├── index.html
│   ├── history.html
│   └── report.html
├── requirements.txt
└── README.md
```

---

## 🔁 System Workflow

1. User uploads an image or video.
2. System applies YOLO model and facial recognition.
3. Objects and faces are detected and annotated.
4. Visual reports are generated.
5. File and analysis are stored and accessible from the history page.

---

## 🔌 API Routes

* `GET /` — Homepage upload form
* `POST /upload` — Process uploaded file and return results
* `GET /history` — View all processed reports
* `GET /uploads/<filename>` — Access processed files

**Example JSON response:**

```json
{
  "success": true,
  "graph_path": "/static/reports/report_graph.png",
  "video_path": "/uploads/processed_video.mp4"
}
```

---

## 🖼️ Image and Video Processing

* **Images:** Processed via OpenCV and YOLO with bounding boxes and face recognition.
* **Videos:** Each frame is processed, saved as MP4, and progress is logged.

---

## 📊 Report Generation

Reports include:

* Bar graphs with detected object counts
* Face presence statistics
* Files saved in `static/reports/`

---

## 🕓 Processing History

* Processed media saved in `uploads/`
* Reports stored in `static/reports/`
* Viewable in the `/history` page

---

## ✅ Project Benefits

* 🧠 Promotes classroom discipline and focus
* 📈 Provides easy-to-read visual analytics
* 💻 Uses accessible, low-cost technology
* 🎓 Helps teachers make data-driven decisions

---

## 💰 Required Investments

* 🧪 Lab infrastructure for deployment and testing
* 🧑‍🏫 Training for staff and educators
* 🤝 Partnerships with schools or universities

---

## 📌 Final Considerations

ClassEye is an innovative solution for educational monitoring using AI. With continuous improvement and academic support, it has the potential to become a powerful tool for modern, data-driven education.

**Developed by:** [Bruno Yudi Kay](https://github.com/brunoxzr)
