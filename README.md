# End-to-End ML Pipeline for Customer Churn Prediction

## Objective
Build a reusable and production-ready machine learning pipeline for predicting customer churn using the Telco Churn Dataset. The pipeline should automate preprocessing, train multiple models, perform hyperparameter tuning, and enable easy deployment.

## Dataset
- **Telco Churn Dataset** (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)  
- Contains customer information such as demographics, account details, services subscribed, and churn status.

## Instructions
- Implement **data preprocessing** (scaling numeric features, encoding categorical features) using Scikit-learn Pipeline  
- Train models like **Logistic Regression** and **Random Forest**  
- Use **GridSearchCV** for hyperparameter tuning  
- Export the complete pipeline using **Joblib**  
- Evaluate models using **accuracy, precision, recall, and F1-score**  

## Workflow
1. Load dataset and separate features/target  
2. Perform train-test split for evaluation  
3. Preprocess numeric (scaling) and categorical (encoding) features  
4. Build pipelines for Logistic Regression and Random Forest  
5. Tune hyperparameters with GridSearchCV  
6. Evaluate both models and compare performance  
7. Export the best pipeline using Joblib  
8. Reload the saved pipeline for new predictions  

## Results

**Logistic Regression**  
- Best Params: `C=1`, `solver=lbfgs`  
- Accuracy: ~83%  
- Stronger recall for “No churn”

**Random Forest**  
- Best Params: `n_estimators=100`, `max_depth=None`  
- Accuracy: ~80%  
- Slightly weaker performance compared to Logistic Regression  

