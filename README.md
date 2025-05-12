# 📘 AI Study Planner

A smart web-based application that helps students plan and analyze their study habits using Artificial Intelligence.  
Built with **Flask**, this project implements all 3 required AI tasks (A, B, C) for the Spring 2025 AI Final Project at IITU.

---

## 🎯 Objective

- Generate a personalized daily study schedule
- Predict academic focus and performance using 8 supervised algorithms
- Analyze study behavior patterns via clustering and dimensionality reduction
- Detect learning-related objects from uploaded images

---

## ✅ Features

| Task | Description |
|------|-------------|
| **Task A** | 8 Supervised Learning Algorithms |
| **Task B** | KMeans Clustering, PCA, FP-Growth |
| **Task C** | Object Detection via image upload |

---

## 🧠 Task A: Supervised Learning

Implemented from scratch:
- Linear Regression
- Logistic Regression
- Decision Tree
- Random Forest
- Naive Bayes
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Gradient Boosting

Each model predicts a student task focus (e.g. Reading, Practice) and returns an accuracy score.

---

## 🧪 Task B: Unsupervised Learning

- **KMeans**: clusters students based on study input
- **PCA**: reduces data for visualization
- **FP-Growth**: finds rules like “Reading → Practice”

---

## 🖼 Task C: Computer Vision

Students can upload an image (e.g. study desk)  
The system simulates object detection and displays the result.

---

## 🖥 How to Run

1. Clone the repo:
```bash
git clone https://github.com/YOUR_USERNAME/ai-study-planner.git
cd ai-study-planner
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
python app.py
```

4. Open in browser:
```
http://localhost:5000
```

---

## 📁 Project Structure

```
ai-study-planner/
│
├── ai_modules/               # All 12 AI algorithms
├── static/
│   ├── style.css             # Pink UI theme
│   └── images/               # Accuracy chart
├── templates/
│   └── index.html            # Web UI
├── app.py                    # Flask app
├── README.md
└── requirements.txt
```

---

## 📷 Screenshots

Add screenshots of prediction results, clustering, and image detection.

---

## 📚 License

This project is for educational use under the IITU AI Final Project Spring 2025.
