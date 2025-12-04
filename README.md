# Material Stream Identification System  
**Machine Learning Course – Cairo University, Faculty of Computers and Artificial Intelligence**

This project implements an **Automated Material Stream Identification (MSI) System** that classifies waste items from live camera input into 7 categories using classical ML techniques (SVM & k-NN). The system follows the full ML pipeline:  
**Data Augmentation → Feature Extraction → Model Training → Real-Time Deployment**.

---

## 📁 Dataset Handling (Important!)

- The dataset is **NOT included** in this repository.
- All team members must manually place the dataset in:

```
data/
├── glass/
├── paper/
├── cardboard/
├── plastic/
├── metal/
└── trash/
```

> 🔒 **Note**: `data/` is ignored via `.gitignore`.

---

## 🧰 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### 1. Data Augmentation  
Generates augmented images and saves them to `final_dataset/`.

```bash
python src/data_augmentation.py
```

### 2. Feature Extraction  
Converts images into numerical feature vectors.

```bash
python src/feature_extraction.py
```

### 3. Train Classifiers  
Trains and saves the SVM and k-NN models.

```bash
python src/train_svm.py
python src/train_knn.py
```

### 4. Real-Time Application  
Runs live camera classification.

```bash
python src/realtime_app.py
```

---

## 🎯 Project Features

- Data Augmentation (rotation, flip, gamma, scaling)
- Balanced dataset (~500 images per class)
- Feature Extraction using 768-D color histograms  
- SVM (RBF) & k-NN (distance-weighted)
- Unknown class rejection if confidence < 80%
- Real-time OpenCV camera deployment  

---

## 📂 Repository Structure

```
material-stream-identification/
├── data/                ← (LOCAL ONLY – NOT TRACKED)
├── models/
├── src/
│   ├── data_augmentation.py
│   ├── feature_extraction.py
│   ├── train_svm.py
│   ├── train_knn.py
│   └── realtime_app.py
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📄 Deliverables (As per Project PDF)

- Source Code Repository  
- Trained Models (`models/*.pkl`)  
- Technical Report (`report.pdf`)
