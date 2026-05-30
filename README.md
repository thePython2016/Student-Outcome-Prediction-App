# Student Outcome Prediction App

A machine learning web application that predicts whether a student is likely to Dropout, Enroll, or Graduate based on academic and demographic data.

The primary focus of the model is identifying students at risk of dropping out and distinguishing them from students likely to graduate.

🔗 **Live Demo:** https://mystudent-outcome-prediction-app.streamlit.app/

---

#  Model Performance

## Overall Metrics

| Metric                 | Score  |
| ---------------------- | ------ |
| Accuracy               | 0.7674 |
| Macro Avg Precision    | 0.7638 |
| Macro Avg Recall       | 0.7111 |
| Macro Avg F1-Score     | 0.7265 |
| Weighted Avg Precision | 0.7900 |
| Weighted Avg Recall    | 0.7674 |
| Weighted Avg F1-Score  | 0.7665 |

---

## Classification Report

| Class    | Precision | Recall | F1-Score | Support |
| -------- | --------- | ------ | -------- | ------- |
| Dropout  | 1.00      | 0.73   | 0.85     | 15      |
| Enrolled | 0.57      | 0.50   | 0.53     | 8       |
| Graduate | 0.72      | 0.90   | 0.80     | 20      |

> The model demonstrates strong performance in identifying students likely to drop out and graduate.
> Precision, Recall, and F1-Score are used as the primary evaluation metrics to measure classification effectiveness across different student outcomes.


##  Model Details

* Algorithm: XGBoost Classifier (`XGBClassifier`)

* Preprocessing: Feature transformation pipeline

* Input Features:

  * Academic performance indicators
  * Demographic information
  * Student behavioral features

* Output: Predicted student outcome

  * Dropout
  * Enrolled
  * Graduate


### 1. Clone the repository

git clone https://github.com/thePython2016/studentOutcomePredictionApp.git
cd studentOutcomePredictionApp


### 2. Install dependencies

pip install -r requirements.txt


### 3. Run the app

streamlit run app.py

##  Sample Data

Use the provided `sample_data.csv` to test the app format.


