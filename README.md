# 🚀 Workflow CI MLflow - Fatalities Model

Project ini merupakan implementasi **Machine Learning Pipeline berbasis MLflow Project** yang terintegrasi dengan **GitHub Actions (CI/CD)** untuk melakukan training model secara otomatis.

---

## 📌 Fitur Utama

- 🔄 **Automated Training**  
  Model akan dilatih ulang setiap kali terjadi push ke branch `main`.

- 📊 **MLflow Tracking**  
  Mencatat parameter, metrics, dan artifacts secara otomatis.

- 📦 **Artifact Management**  
  Menyimpan:
  - Model (`model.pkl`)
  - Classification report
  - MLflow model format

- 🐳 **Docker Integration (Advance)**  
  Model dibuild menjadi Docker Image dan di-push ke Docker Hub.

---

## 🗂️ Struktur Project

```bash
Workflow-CI
├── .github/workflows/
│ └── ci.yml
├── MLProject/
│ ├── modelling.py
│ ├── MLproject
│ ├── conda.yaml
│ ├── fatalities_preprocessed/
│ └── outputs/
└── README.md
```

---

## ⚙️ Cara Menjalankan Secara Manual

```bash
cd MLProject
mlflow run . -e train
```

Atau dengan parameter:

```bash
mlflow run . -e train -P max_iter=5000 -P solver=lbfgs -P class_weight=balanced
```

---

## 🔁 Workflow CI/CD

Workflow akan berjalan otomatis ketika:

- Push ke branch main
- Manual trigger dari GitHub Actions (workflow_dispatch)

Proses yang dilakukan:

1. Checkout repository
2. Setup Python environment
3. Menjalankan MLflow Project
4. Menyimpan artifact model
5. Build Docker image
6. Push ke Docker Hub

---

## 🐳 Docker Hub

Docker image model tersedia di:

👉 <https://hub.docker.com/r/USERNAME/fatalities-model>

---

## 📊 Output yang Dihasilkan

- Model Machine Learning (model.pkl)
- Classification Report (classification_report.txt)
- MLflow logged model
- Docker image siap deploy

---

## 🧠 Teknologi yang Digunakan

- Python 3.12
- Scikit-learn
- MLflow
- GitHub Actions
- Docker

---

## ✨ Catatan

Project ini dirancang mengikuti praktik terbaik dalam:

- Reproducible ML pipeline
- CI/CD untuk Machine Learning
- Model packaging & deployment

---
