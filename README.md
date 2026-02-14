## a.Problem Statement

# 🎈 Binary Classification - Comparison App

This app provides ability to perform and compare various Binary classification methods. below models are already implemented and ready for testing 

1. Logistic Regression
2. Decision Tree Classifi er
3. K-Nearest Neighbor Classifi er
4. Naive Bayes Classifi er - Gaussian or Multinomial
5. Ensemble Model - Random Forest
6. Ensemble Model - XGBoost


Metrics like 1. Accuracy 2. Precision 3. Recall 4. AUC  5. F1 Score and 6. MCC score are evaluated 

Besides this page will also display corresponding Confusion Matrix

### https://blank-app-293qe31bv78.streamlit.app/


3. Download Test data from here /or the app page itself  -https://raw.githubusercontent.com/digidetoxpr-eng/blank-app/refs/heads/main/Airline_customer_satisfaction.csv

4. upload the Test Data in the app and choose required binary classification method. click on "Run" button to get classficiation metrics.

## b.Data set Description 

we are using a Airline customer service survey data to predict customer satisfaction levels ( Satisfied /Un Satisfied) . This is a binary classfication problem.
Data downloaded from Kaggle having 22 columns and 129880 rows of data

Feature Size:22
Instance Size:129880




## c.Models Used 

1. Logistic Regression
2. Decision Tree Classifi er
3. K-Nearest Neighbor Classifi er
4. Naive Bayes Classifi er - Gaussian or Multinomial
5. Ensemble Model - Random Forest
6. Ensemble Model - XGBoost



| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.7612 | 0.7525 | 0.7551 | 0.83805 | 0.7944 | 0.5153 |
| Decision Tree | 0.8471 | 0.83749 | 0.8157 | 0.9331 | 0.8705 | 0.6953 |
| KNN | 0.7265 | 0.7234 | 0.7503 | 0.7541 | 0.7522 | 0.4470 |
| Naïve Bayes | 0.8206 | 0.8187 | 0.8368 | 0.8376 | 0.8372 | 0.6376 |
| Random Forest (Ensemble) | 0.9571 | 0.9578 | 0.9707 | 0.9507 | 0.9606 | 0.9137 |
| XGBoost (Ensemble) | 0.9045 | 0.9020 | 0.9021 | 0.9272 | 0.9145 | 0.8069 |



| ML Model Name | Model Performance |
| :--- | :--- |
| Logistic Regression | Logistic Regression is a moderate-performing baseline model. It demonstrates a strong ability to identify positive cases (Recall: 0.838) but has a lower overall accuracy (0.761) compared to more complex ensemble methods. it significantly lags behind ensemble models like Random Forest |
| Decision Tree | While the Decision Tree has lower overall accuracy, it shows a very high Recall (0.9331), indicating it is effective at identifying positive cases, though at the cost of more false positives. |
| KNN | KNN scored the lowest in Accuracy (0.7265) and Matthews Correlation Coefficient (MCC) (0.4470), the latter of which is often used to assess quality in imbalanced datasets |
| Naïve Bayes | maintains very stable scores across Accuracy, Precision, Recall, and F1 (all ~0.82–0.83), showing consistent behavior across different evaluation dimensions. |
| Random Forest(Ensemble) | Random Forest leads in nearly every metric, particularly in Precision (0.9707), suggesting it has the lowest rate of false positives. |
| XGBoost (Ensemble) | XGBoost (Ensemble) is a high-performing model that ranks second overall among the evaluated algorithms. It demonstrates strong balanced performance across all classification metrics. As a gradient boosting algorithm, XGBoost improves by building an ensemble of decision trees sequentially, where each new tree corrects the errors of previous ones. |






