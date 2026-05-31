# 🎓 Student Outcome Prediction App

A machine learning web application that predicts whether a student is likely to **Dropout**, **Enroll**, or **Graduate** based on academic and demographic data.

The primary focus of the model is identifying students at risk of dropping out and distinguishing them from students likely to graduate.

🔗 **Live Demo:** https://mystudent-outcome-prediction-app.streamlit.app/

---

# 📊 Model Performance

## Overall Metrics

| Metric | Score |
| ---------------------- | ------ |
| Accuracy | 0.7470 |
| Macro Avg Precision | 0.7159 |
| Macro Avg Recall | 0.7186 |
| Macro Avg F1-Score | 0.7119 |
| Weighted Avg Precision | 0.7708 |
| Weighted Avg Recall | 0.7470 |
| Weighted Avg F1-Score | 0.7551 |

---

## Classification Report

| Class | Precision | Recall | F1-Score | Support |
| -------- | --------- | ------ | -------- | ------- |
| Dropout | 0.8291 | 0.7150 | 0.7678 | 414 |
| Enrolled | 0.4690 | 0.6285 | 0.5372 | 253 |
| Graduate | 0.8497 | 0.8124 | 0.8306 | 661 |

> The model demonstrates strong performance in identifying students likely to graduate and students at risk of dropping out. The Dropout class achieves a Precision of 0.8291, Recall of 0.7150, and F1-Score of 0.7678, while the Graduate class achieves a Precision of 0.8497, Recall of 0.8124, and F1-Score of 0.8306. Precision, Recall, and F1-Score are used as the primary evaluation metrics to measure classification effectiveness across different student outcomes.
> 
## Model Detail

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



## 📜 License

This project is open source and available under the MIT License.
