import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="centered"
)


# =========================================================
# TITLE
# =========================================================

st.title("🏦 Customer Churn Prediction")

st.write(
    "Enter customer information and predict whether the customer will stay or not stay."
)

st.divider()


# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("churn_clean_Data.csv")


# =========================================================
# ONLY THESE 10 FEATURES
# =========================================================

features = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary"
]


X = df[features]

# Target
# 0 = Stay
# 1 = Not Stay / Churn
y = df["Exited"]


# =========================================================
# NUMERIC & CATEGORICAL COLUMNS
# =========================================================

numeric_columns = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary"
]


categorical_columns = [
    "Geography",
    "Gender"
]


# =========================================================
# PREPROCESSING
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[

        (
            "num",
            StandardScaler(),
            numeric_columns
        ),

        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ]
)


# =========================================================
# MODEL
# =========================================================

model = Pipeline(
    steps=[

        (
            "preprocessing",
            preprocessor
        ),

        (
            "model",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000
            )
        )
    ]
)


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# TRAIN MODEL
# =========================================================

model.fit(X_train, y_train)


# =========================================================
# CUSTOMER INFORMATION
# =========================================================

st.subheader("👤 Customer Information")


col1, col2 = st.columns(2)


# =========================================================
# LEFT COLUMN
# =========================================================

with col1:

    CreditScore = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=650,
        step=1
    )


    Geography = st.selectbox(
        "Geography",
        ["France", "Spain", "Germany"]
    )


    Gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )


    Age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=40,
        step=1
    )


    Tenure = st.number_input(
        "Tenure",
        min_value=0,
        max_value=10,
        value=5,
        step=1
    )


# =========================================================
# RIGHT COLUMN
# =========================================================

with col2:

    Balance = st.number_input(
        "Balance",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )


    NumOfProducts = st.number_input(
        "Number of Products",
        min_value=1,
        max_value=4,
        value=1,
        step=1
    )


    # IMPORTANT:
    # User enters 0 or 1 directly

    HasCrCard = st.number_input(
        "Has Credit Card (0 = No, 1 = Yes)",
        min_value=0,
        max_value=1,
        value=1,
        step=1
    )


    # IMPORTANT:
    # User enters 0 or 1 directly

    IsActiveMember = st.number_input(
        "Active Member (0 = No, 1 = Yes)",
        min_value=0,
        max_value=1,
        value=1,
        step=1
    )


    EstimatedSalary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=100000.0,
        step=1000.0
    )


# =========================================================
# PREDICT BUTTON
# =========================================================

st.write("")

predict_button = st.button(
    "🔮 Predict Churn",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # Create customer dataframe
    customer = pd.DataFrame({

        "CreditScore": [CreditScore],

        "Geography": [Geography],

        "Gender": [Gender],

        "Age": [Age],

        "Tenure": [Tenure],

        "Balance": [Balance],

        "NumOfProducts": [NumOfProducts],

        "HasCrCard": [HasCrCard],

        "IsActiveMember": [IsActiveMember],

        "EstimatedSalary": [EstimatedSalary]
    })


    # =====================================================
    # PREDICT 0 OR 1
    # =====================================================

    prediction = int(model.predict(customer)[0])


    # =====================================================
    # PROBABILITIES
    # =====================================================

    probabilities = model.predict_proba(customer)[0]

    stay_probability = probabilities[0]

    not_stay_probability = probabilities[1]


    # =====================================================
    # RESULT
    # =====================================================

    st.divider()

    st.subheader("📊 Prediction Result")


    # -----------------------------------------------------
    # STAY
    # -----------------------------------------------------

    if prediction == 0:

        st.success(
            "✅ Prediction: 0 — STAY"
        )

        st.write(
            f"**Stay Probability: {stay_probability:.2%}**"
        )

        st.write(
            f"**Not Stay Probability: {not_stay_probability:.2%}**"
        )


    # -----------------------------------------------------
    # NOT STAY / CHURN
    # -----------------------------------------------------

    else:

        st.error(
            "⚠️ Prediction: 1 — NOT STAY / CHURN"
        )

        st.write(
            f"**Stay Probability: {stay_probability:.2%}**"
        )

        st.write(
            f"**Not Stay Probability: {not_stay_probability:.2%}**"
        )