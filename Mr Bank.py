# Mr Bank
# AI ChatBot V4

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random

#Load and pepare dataset
DATA_PATH = r"C:\Users\adamm\Downloads\train.csv"

def load_and_train_model():
    df = pd.read_csv(DATA_PATH)

    # Handle missing values
    df.fillna(df.mode().iloc[0], inplace=True)

    # Encode categorical variables
    categorical_columns = [
        "Gender", "Married", "Dependents", "Education",
        "Employment_Status", "Property_Area", "Loan_Status"
    ]

    le = LabelEncoder()
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])

    # Feature selection
    X = df[[
        "Applicant_Income", "Coapplicant_Income", "Loan_Amount",
        "Loan_Term", "Credit_History", "Education",
        "Employment_Status", "Property_Area", "Age"
    ]]
    y = df["Loan_Status"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Model trained successfully. Accuracy: {accuracy:.2f}")

    return model, le

#Chatbot logic
HINTS = [
    "try saying 'check eligibility'",
    "ask about 'loan approval'",
    "say 'credit history info'",
    "type 'bye' to exit"
]

def predict_eligibility(model):
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

        # Predict
        features = [[income, co_income, loan_amt, loan_term, credit_hist,
                     education, employment, property_area, age]]
        prediction = model.predict(features)[0]

        result = "Approved ✅" if prediction == 1 else "Not Approved ❌"
        print(f"\nPrediction: {result}")
        if prediction == 1:
            print("Based on your details, you meet our estimated loan approval criteria.")
        else:
            print("Your current details suggest loan approval is less likely.")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure all values are entered correctly.\n")

def get_response(user_input, model):
    user_input = user_input.lower().strip()

    if "hello" in user_input or "hi" in user_input:
        return "Hello! I can help predict your loan eligibility. Type 'check eligibility' to begin."
    elif "loan" in user_input or "eligibility" in user_input:
        predict_eligibility(model)
        return "Would you like to check another loan?"
    elif "credit" in user_input:
        return "A positive credit history (1) improves your loan approval chances."
    elif "bye" in user_input or "exit" in user_input:
        return "Goodbye! Have a great day."
    else:
        hint = random.choice(HINTS)
        return f"I'm not sure how to respond to that yet, try saying {hint}."

#Main
def main():
    print("AI Chatbot v3.0 — Loan Eligibility Assistant")

    model, _ = load_and_train_model()

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["bye", "exit", "quit"]:
            print("Chatbot: Goodbye!")
            break

        response = get_response(user_input, model)
        if response:
            print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()