# Mr Bank
# AI ChatBot V4

# Fixed convergence issues, added scaling and better error handling

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random
import warnings

warnings.filterwarnings("ignore")  # Hide convergence and warning messages

# -------------------- Load and prepare dataset --------------------
DATA_PATH = r"C:\Users\adamm\Downloads\train.csv"

def load_and_train_model():
    df = pd.read_csv(DATA_PATH)

    # Fill missing values with mode
    df.fillna(df.mode().iloc[0], inplace=True)

    # Encode catgorical variables
    categorical_columns = [
        "Gender", "Married", "Dependents", "Education",
        "Employment_Status", "Property_Area", "Loan_Status"
    ]

    le = LabelEncoder()
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col].astype(str))

    # Features and target
    X = df[[
        "Applicant_Income", "Coapplicant_Income", "Loan_Amount",
        "Loan_Term", "Credit_History", "Education",
        "Employment_Status", "Property_Area", "Age"
    ]]
    y = df["Loan_Status"]

    # Scale numeric data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train model with higher iteration limit
    model = LogisticRegression(max_iter=2000, solver='lbfgs')
    model.fit(X_train, y_train)

    # Evaluate
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Model trained successfully. Accuracy: {accuracy:.2f}")

    return model, scaler

# -------------------- Chatbot logic --------------------
HINTS = [
    "try saying 'check eligibility'",
    "ask about 'loan approval'",
    "say 'credit history info'",
    "type 'bye' to exit"
]

def predict_eligibility(model, scaler):
    try:
        print("\nLet's check your loan eligibility.")
        income = float(input("Enter your monthly income (£): "))
        co_income = float(input("Enter your coapplicant's income (£): "))
        loan_amt = float(input("Enter desired loan amount (£): "))
        loan_term = float(input("Enter loan term in months: "))
        credit_hist = int(input("Do you have a positive credit history? (1 = Yes, 0 = No): "))
        education = int(input("Are you a graduate? (1 = Yes, 0 = No): "))
        employment = int(input("Are you employed? (1 = Yes, 0 = No): "))
        property_area = int(input("Property area (0 = Rural, 1 = Semiurban, 2 = Urban): "))
        age = int(input("Enter your age: "))

        # Validate input ranges
        if any(v < 0 for v in [income, co_income, loan_amt, loan_term, age]):
            print("Invalid input. Values cannot be negative.\n")
            return

        # Prepare data and scale
        features = np.array([[income, co_income, loan_amt, loan_term, credit_hist,
                              education, employment, property_area, age]])
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]
        result = "Approved ✅" if prediction == 1 else "Not Approved ❌"
        print(f"\nPrediction: {result}")
        if prediction == 1:
            print("Based on your financial details, loan approval is likely.")
        else:
            print("Loan approval may not be granted based on your current data.")
        print()
    except ValueError:
        print("Invalid input type. Please enter numeric values only.\n")
    except Exception as e:
        print(f"Unexpected error: {e}\n")

def get_response(user_input, model, scaler):
    user_input = user_input.lower().strip()

    if "hello" in user_input or "hi" in user_input:
        return "Hello! I can help predict your loan eligibility. Type 'check eligibility' to begin."
    elif "loan" in user_input or "eligibility" in user_input:
        predict_eligibility(model, scaler)
        return "Would you like to check another loan?"
    elif "credit" in user_input:
        return "A positive credit history (1) increases your likelihood of loan approval."
    elif "bye" in user_input or "exit" in user_input:
        return "Goodbye! Have a great day."
    else:
        hint = random.choice(HINTS)
        return f"I'm not sure how to respond to that yet, try saying {hint}."

# -------------------- Main --------------------
def main():
    print("AI Chatbot v5.0 — Loan Eligibility Assistant (Optimised)")
    print("Now with scaled data and improved model convergence.\n")

    model, scaler = load_and_train_model()

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["bye", "exit", "quit"]:
            print("Chatbot: Goodbye!")
            break

        response = get_response(user_input, model, scaler)
        if response:
            print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()