# Customer Churn Prediction Using Machine Learning

## 📌 Project Overview

Customer Churn Prediction is a Machine Learning project that predicts whether a customer is likely to leave a bank or continue using its services.

Customer churn is a major challenge for businesses because losing existing customers can affect revenue and business growth.

In this project, I developed and evaluated multiple Machine Learning classification algorithms to predict customer churn based on customer information such as credit score, age, geography, gender, account balance, tenure, and other relevant features.

The project also uses a Machine Learning pipeline to perform preprocessing and prediction efficiently.

## 🎯 Project Objectives

- Analyze customer data to understand churn patterns.
- Perform Exploratory Data Analysis (EDA).
- Handle categorical features using encoding techniques.
- Apply feature scaling to numerical data.
- Build and compare different Machine Learning models.
- Evaluate model performance using classification metrics.
- Predict whether a new customer will leave the bank.
- Save the trained pipeline for future predictions.

## 📂 Dataset Information

The project uses a customer churn dataset (`churn_clean_Data.csv`).

**Target Variable:** `Exited`

- `0` – Customer did not leave the bank.
- `1` – Customer left the bank.

### Dataset Features

| Feature | Description |
|---|---|
| RowNumber | Record number |
| CustomerId | Unique customer ID |
| Surname | Customer surname |
| CreditScore | Customer credit score |
| Geography | Customer's country |
| Gender | Customer's gender |
| Age | Customer's age |
| Tenure | Number of years with the bank |
| Balance | Customer's account balance |
| NumOfProducts | Number of bank products used |
| HasCrCard | Whether customer has a credit card |
| IsActiveMember | Whether customer is an active member |
| EstimatedSalary | Customer's estimated salary |
| Exited | Target variable indicating customer churn |

**Unnecessary Features Removed:**
- `RowNumber`
- `CustomerId`
- `Surname`

These columns were removed because they do not provide useful predictive information for the model.

## 🛠️ Technologies and Libraries Used

- Python
- Jupyter Notebook
- Pandas – Data manipulation and analysis
- NumPy – Numerical operations
- Matplotlib – Data visualization
- Seaborn – Statistical data visualization
- Scikit-learn – Machine Learning and model evaluation
- Joblib – Saving the trained ML pipeline

## 🔍 Project Workflow

1. **Data Collection:** Load the customer churn dataset using Pandas.
2. **Data Exploration:** Understand dataset structure, dimensions, data types, and target distribution.
3. **Data Cleaning:** Check missing values and duplicate records.
4. **Exploratory Data Analysis:** Analyze numerical features and visualize correlations using a heatmap.
5. **Feature Selection:** Separate input features and target variable. Remove unnecessary columns.
6. **Data Preprocessing:** Encode categorical features and scale numerical features.
7. **Train-Test Split:** Divide the dataset into 80% training data and 20% testing data.
8. **Model Building:** Train different classification algorithms.
9. **Model Evaluation:** Evaluate models using accuracy and classification reports.
10. **Pipeline Creation:** Combine preprocessing and Logistic Regression into a single pipeline.
11. **Prediction:** Predict churn for new customer information.
12. **Model Saving:** Save the trained pipeline using Joblib.

## 🤖 Machine Learning Algorithms Used

The following algorithms were implemented and evaluated:

1. **Logistic Regression** – Used for binary classification to predict customer churn.
2. **Decision Tree Classifier** – Makes predictions using decision rules.
3. **Naive Bayes Classifier** – Uses probability-based classification.
4. **K-Nearest Neighbors (KNN)** – Predicts outcomes based on the nearest data points.

### Final Pipeline

A Scikit-learn Pipeline was created using:

- StandardScaler – Scaling numerical features.
- OneHotEncoder – Encoding categorical features.
- LogisticRegression – Final classification model with balanced class weights.

## 📊 Model Evaluation

The models were evaluated using the following metrics:

- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

These metrics help evaluate how effectively the models identify customers who are likely to leave the bank.

## 🔮 Sample Prediction

A sample customer record was created with the following details:

- Credit Score: 619
- Geography: France
- Gender: Female
- Age: 42
- Tenure: 2
- Balance: 0.00
- Number of Products: 1
- Has Credit Card: 1
- Active Member: 1
- Estimated Salary: 101348.88

The trained pipeline predicts whether this customer is likely to leave the bank.

The `predict_proba()` method is also used to obtain prediction probabilities.

## 💾 Model Deployment Preparation

The trained Machine Learning pipeline is saved using Joblib.

```python
import joblib

joblib.dump(pipeline, "churn_pipeline.pkl")
```

Load the saved model:

```python
import joblib

model = joblib.load("churn_pipeline.pkl")
```

The saved pipeline can be reused for predictions without manually performing preprocessing again.

## 📁 Project Structure

```text
Customer-Churn-Prediction/
│
├── Churn_Model.ipynb
├── churn_clean_Data.csv
├── churn_pipeline.pkl
├── README.md
└── requirements.txt
```

## ⚙️ Installation and Execution

**Step 1: Clone the repository**

```bash
git clone https://github.com/your-username/Customer-Churn-Prediction.git
```

**Step 2: Navigate to the project directory**

```bash
cd Customer-Churn-Prediction
```

**Step 3: Install required libraries**

```bash
pip install -r requirements.txt
```

**Step 4: Open Jupyter Notebook**

```bash
jupyter notebook
```

**Step 5:** Open `Churn_Model.ipynb` and run the cells to explore the data, train the models, and make predictions.

## 📚 Requirements

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
jupyter
```

## 🚀 Future Improvements

- Develop an interactive web application using Streamlit.
- Improve model performance through hyperparameter tuning.
- Apply feature selection techniques.
- Compare additional Machine Learning algorithms.
- Deploy the application on Streamlit Community Cloud.
- Provide an easy-to-use interface for predicting customer churn.

## 👩‍💻 Author

**Laxmi Rahangdale**

B.Tech – Computer Science and Engineering  
Specialization: Artificial Intelligence and Machine Learning

**Technical Skills:** Python | C++ | SQL | Machine Learning | Data Science | Power BI

---

⭐ If you find this project useful, don't forget to star the repository!


