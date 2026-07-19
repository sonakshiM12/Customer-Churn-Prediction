# 📊 Customer Churn Prediction Using Machine Learning

## 📌 Project Overview

Customer churn prediction is the process of identifying customers who are likely to discontinue a company's services.

This project uses a Machine Learning model to predict customer churn based on customer information. A Streamlit web application provides an easy-to-use interface for making predictions.

---

## 🚀 Features

- Customer churn prediction
- Random Forest Machine Learning model
- Streamlit web interface
- Data preprocessing
- Label Encoding
- Model accuracy evaluation
- Feature importance analysis
- Probability prediction

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

---

## 📂 Project Structure

```
Customer-Churn-Prediction/
│
├── app.py
├── train.py
├── data/
│   └── churn.csv
│
├── model/
│   ├── churn_model.pkl
│   ├── encoders.pkl
│   └── features.pkl
│
├── README.md
├── LICENSE
├── requirements.txt
└── .gitignore
```

---

## 📊 Machine Learning Workflow

1. Load Dataset
2. Data Cleaning
3. Data Preprocessing
4. Label Encoding
5. Train-Test Split
6. Random Forest Training
7. Model Evaluation
8. Save Model
9. Streamlit Deployment

---

## 📈 Model Performance

Algorithm Used:

- Random Forest Classifier

Accuracy:

**79.56%**

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 💻 How to Run

Clone the repository

```bash
git clone https://github.com/sonakshiM12/Customer-Churn-Prediction.git
```

Move inside project

```bash
cd Customer-Churn-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Train the model

```bash
python train.py
```

Run the application

```bash
streamlit run app.py
```

---

## 📸 Application

The Streamlit application allows users to enter customer information and instantly predict whether the customer is likely to churn.

---

## 👩‍💻 Author

**Sonakshi Mishra**

B.Tech Computer Science (AI & ML)

---
## Screenshots
### Home Page


## 📜 License

This project is licensed under the MIT License.
