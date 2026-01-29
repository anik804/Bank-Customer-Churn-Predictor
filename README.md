# 🏦 Bank Customer Churn Prediction

A machine learning project designed to predict whether a bank customer is likely to leave (churn) based on their profile and financial status. This project features a high-performance **Gradient Boosting** model and a modern, interactive **Gradio** web interface.

## 🚀 Overview

Predicting customer churn is critical for banks to maintain their customer base. This application allows bank analysts to input specific customer data—such as credit score, balance, and account activity—and receive an instant prediction on the likelihood of that customer exiting.

## 🛡️ Key Features

- **Predictive Analytics**: Uses a Gradient Boosting Classifier to achieve high accuracy.
- **Modern UI**: A sleek, dark-themed Gradio web interface for easy interaction.
- **Robust Preprocessing**: Automated handling of categorical data (Geography, Gender) and numerical scaling.
- **Real-time Results**: Instant churn risk assessment with associated probability/confidence scores.

## 🛠️ Tech Stack

- **Language**: Python 3.x
- **Machine Learning**: `scikit-learn` (Gradient Boosting, Pipelines, Preprocessing)
- **Data Manipulation**: `pandas`, `numpy`
- **Web Interface**: `gradio`
- **Model Storage**: `pickle`

## 📊 Dataset Detail

The model is trained on the **Churn_Modelling.csv** dataset, which includes:
- **Demographics**: Geography, Gender, Age.
- **Account Details**: Tenure (years with bank), Balance, Number of Products.
- **Behavioral Data**: Has Credit Card, Is Active Member.
- **Financial Status**: Credit Score, Estimated Salary.

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model (Optional)
If you wish to retrain the model from scratch:
```bash
python gb_train.py
```

### 4. Run the Web Application
Launch the interactive Gradio dashboard:
```bash
python app.py
```
After running, visit `http://127.0.0.1:7860` in your browser.

## 📂 Project Structure

- `app.py`: The Main web application entry point (Gradio UI).
- `gb_train.py`: Training script specifically for the Gradient Boosting model.
- `rf_train.py`: Alternative training script for Random Forest.
- `Churn_Modelling.csv`: The core dataset.
- `gb_churn_model.pkl`: The serialized model pipeline (preprocessor + classifier).
- `requirements.txt`: List of required Python packages.

---
