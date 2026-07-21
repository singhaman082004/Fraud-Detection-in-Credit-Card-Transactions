import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Dataset

data = {
    "Amount": [100, 5000, 200, 10000, 150, 7000, 300, 12000, 250, 8000,
               180, 9500, 220, 11000, 170, 6500, 280, 10500, 190, 8500],

    "Time": [10, 2, 15, 1, 20, 3, 25, 2, 18, 1,
             22, 2, 14, 1, 21, 4, 16, 2, 19, 3],

    "Transactions_Per_Day": [2, 20, 3, 25, 2, 18, 4, 30, 3, 22,
                             2, 24, 3, 28, 2, 17, 4, 27, 3, 21],

    "Fraud": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
              0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Features and Target

X = df.drop("Fraud", axis=1)
y = df["Fraud"]

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model Training

model = LogisticRegression()
model.fit(X_train, y_train)


# Evaluation

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Streamlit UI

st.title("Credit Card Fraud Detection")

st.write("Enter Transaction Details")

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=1000.0
)

time = st.number_input(
    "Transaction Time (Hour)",
    min_value=0,
    max_value=23,
    value=12
)

transactions = st.number_input(
    "Transactions Per Day",
    min_value=0,
    value=5
)

if st.button("Predict Fraud"):

    prediction = model.predict(
        [[amount, time, transactions]]
    )

    if prediction[0] == 1:
        st.error("Fraud Transaction Detected")
    else:
        st.success("Genuine Transaction")


# Model Performance

st.subheader("Model Accuracy")
st.write(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix Table

st.subheader("Confusion Matrix")

cm_df = pd.DataFrame(
    cm,
    index=["Actual Genuine", "Actual Fraud"],
    columns=["Predicted Genuine", "Predicted Fraud"]
)

st.dataframe(cm_df)


# Confusion Matrix Graph

fig, ax = plt.subplots()

ax.imshow(cm)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(["Genuine", "Fraud"])
ax.set_yticklabels(["Genuine", "Fraud"])

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

st.pyplot(fig)

# Classification Report

st.subheader("Classification Report")
st.text(report)

# Dataset View

st.subheader("Dataset")
st.dataframe(df)
