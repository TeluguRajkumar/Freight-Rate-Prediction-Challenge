 🚚 Freight Rate Prediction using Machine Learning

## 📌 Overview

This project develops a machine learning regression model to predict freight transportation rates using historical shipment data. The solution includes data preprocessing, feature engineering, model training, validation, and prediction generation for unseen freight loads.

The trained model predicts freight rates for:
- Validation dataset (`validation.csv`)
- December shipment dataset (`december_chart_inputs.csv`)

The generated prediction files follow the required submission format and are compatible with the provided scoring script.

---

## 🎯 Objectives

- Build an accurate freight rate prediction model.
- Perform data preprocessing and feature engineering.
- Train and validate regression models.
- Generate predictions for unseen shipment data.
- Produce submission files in the required format.

---

# 🚚 Freight Rate Prediction using Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-success)
![License](https://img.shields.io/badge/License-MIT-green)

**A Production-Ready Machine Learning Pipeline for Predicting Freight Transportation Rates**

</div>

---

## 🎯 Problem Statement

Develop a machine learning regression model to predict freight transportation rates using historical shipment data. Train and validate the model on labeled data, then generate accurate predictions for unseen freight loads and produce submission-ready prediction files.

---

## ✨ Features

- 📊 Exploratory Data Analysis (EDA)
- 🧹 Data Cleaning & Preprocessing
- ⚙️ Feature Engineering
- 🤖 Regression Model Training
- 📈 Cross Validation
- 📉 Model Evaluation
- 🔮 Freight Rate Prediction
- 📦 Automated Submission File Generation
- 🚀 Modular & Scalable Codebase

---

## 📂 Dataset

| Dataset | Description |
|----------|-------------|
| `train_test.csv` | Historical shipment data for training |
| `validation.csv` | Unseen shipment data for prediction |
| `validation_predictions_template.csv` | Prediction submission template |
| `december_chart_inputs.csv` | December shipment prediction dataset |

---

## 🛠️ Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

---

## 📁 Project Structure

```text
Freight-Rate-Prediction/
│
├── data/
├── models/
├── outputs/
├── src/
├── tests/
├── train.py
├── predict.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## 🔄 Machine Learning Workflow

```text
Historical Data
       │
       ▼
Data Cleaning
       │
       ▼
Feature Engineering
       │
       ▼
Model Training
       │
       ▼
Model Evaluation
       │
       ▼
Prediction Generation
       │
       ▼
Submission Files
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/<your-github-username>/Freight-Rate-Prediction.git
cd Freight-Rate-Prediction
```

Install the required dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

```bash
python train.py
```

---

## 🔮 Generate Predictions

```bash
python predict.py
```

Generated output files:

- `validation_predictions.csv`
- Updated `december_chart_inputs.csv`

---

## 📊 Evaluation

Validate the generated prediction files using:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

---

## 📈 Future Improvements

- CatBoost Regression
- XGBoost Integration
- LightGBM Support
- Hyperparameter Optimization
- SHAP Explainability
- MLflow Experiment Tracking
- Docker Containerization
- FastAPI Deployment

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

Feel free to fork this repository and submit a pull request.

---

## 👨‍💻 Author

**Rajkumar**

- 💼 Data Science | Machine Learning | Artificial Intelligence
- 🔗 GitHub: https://github.com/TeluguRajkumar
- 🔗 LinkedIn: https://www.linkedin.com/in/raj-kumar-34077a148/

---

## ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub.

---

## 📜 License

This project is licensed under the **MIT License**.
