# Student Outcome Prediction App

A machine learning web application designed to predict whether a student is likely to **Dropout, Enroll, or Graduate** based on academic and demographic data, with primary focus on identifying students at risk of dropping out and those likely to graduate.

🔗 Live App: https://mystudent-outcome-prediction-app.streamlit.app/

---

## Model Performance

| Class    | Precision | Recall | F1-Score |
| -------- | --------- | ------ | -------- |
| Dropout  | 1.00      | 0.73   | 0.85     |
| Enrolled | 0.57      | 0.50   | 0.53     |
| Graduate | 0.72      | 0.90   | 0.80     |

| Metric          | Score |
| --------------- | ----- |
| Accuracy        | 0.767 |
| Macro Avg F1    | 0.726 |
| Weighted Avg F1 | 0.766 |

> The primary goal of this project is to identify students at risk of dropping out and distinguish them from students likely to graduate.
> The model shows strong performance in detecting dropout and graduate outcomes using precision, recall, and F1-score metrics.



##  Features

* 📂 Upload CSV file with student data
* 🔍 Predict student outcomes (Dropout, Enrolled, Graduate)
* 🎯 Main focus on identifying Dropout and Graduate students
* 📊 Displays prediction results instantly
* ⬇️ Download predictions as CSV
* 🤖 Powered by an XGBoost machine learning classification model

---


##  Model Details

* Algorithm: XGBoost Classifier (`XGBClassifier`)

* Frameworks & Libraries:

  * Scikit-learn
  * XGBoost
  * Pandas
  * NumPy

* Target Classes:

  * Dropout
  * Enrolled
  * Graduate

* Evaluation Metrics:

  * Precision
  * Recall
  * F1-score
  * Accuracy

* Primary Evaluation Focus:

  * Dropout detection 
  * Graduate prediction performance



##  Run Locally

### 1. Clone the repository


git clone https://github.com/thePython2016/studentScorePredictionModel.git

cd studentScorePredictionModel


###  Install dependencies

pip install -r requirements.txt


### 3. Run the app

streamlit run app.py

## 📥 Sample Data

Use `Sample data to test.csv` to test the application and understand the expected input format.


## 📋 Requirements

See `requirements.txt` for full dependencies.

## 📌 How to Use

1. Open the live app
2. Upload a CSV file (use sample format)
3. Click **Predict**
4. View predicted student outcomes
5. Download results as CSV




