# Mr Bank
# AI ChatBot V4

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# === Load and prepare data ===
data_path = r"C:\Users\adamm\Downloads\train.csv"
data = pd.read_csv(data_path).dropna()

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Split data
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

# Scale and train model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=2000, solver='lbfgs')
model.fit(X_train, y_train)

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model trained successfully. Accuracy: {accuracy:.2f}\n")

# === Chatbot Questions ===
questions = [
    ("Gender", "What is your gender? (Male/Female)"),
    ("Married", "Are you married? (Yes/No)"),
    ("Dependents", "How many dependents do you have? (0, 1, 2, 3+)"),
    ("Education", "What is your education level? (Graduate/Not Graduate)"),
    ("Employment_Status", "Are you self-employed or salaried? (Self_Employed/Salaried)"),
    ("Applicant_Income", "What is your monthly income?"),
    ("Coapplicant_Income", "What is your co-applicant’s monthly income? (Enter 0 if none)"),
    ("Loan_Amount", "What loan amount are you requesting?"),
    ("Loan_Term", "Over how many months do you plan to repay? (e.g., 360)"),
    ("Credit_History", "Do you have a positive credit history? (1 for Yes, 0 for No)"),
    ("Property_Area", "What type of property area? (Urban/Semiurban/Rural)"),
    ("Age", "What is your age?")
]

# === Prediction Logic ===
def predict_loan_eligibility(user_data):
    try:
        user_df = pd.DataFrame([user_data], columns=X.columns)
        for col in user_df.columns:
            if col in label_encoders:
                try:
                    user_df[col] = label_encoders[col].transform(user_df[col])
                except ValueError:
                    return f"Mr Bank: I don't recognise your answer for '{col}'. Please try again."
            else:
                user_df[col] = user_df[col].astype(float)

        user_scaled = scaler.transform(user_df)
        prediction = model.predict(user_scaled)[0]
        if prediction == 1:
            return "Mr Bank: Congratulations! Based on your details, your loan is likely to be approved."
        else:
            return "Mr Bank: Unfortunately, your loan eligibility seems low. Consider improving your credit history or income ratio."
    except Exception as e:
        return f"Mr Bank: Oops, there was an issue processing your input — {str(e)}"

# === Chatbot Flow ===
print("Welcome to Mr Bank — your AI Loan Assistant (v7.0)")
print("Type 'loan' to start a loan eligibility check or 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip().lower()

    if user_input == "exit":
        print("Mr Bank: Thank you for visiting. Goodbye!")
        break

    elif user_input == "loan":
        responses = {}
        for col, question in questions:
            answer = input(f"Mr Bank: {question}\nYou: ").strip()
            responses[col] = answer

        result = predict_loan_eligibility(responses)
        print(result)
        print("\nMr Bank: Type 'loan' to check another application or 'exit' to leave.\n")

    else:
        print("Mr Bank: I'm not sure how to respond to that yet. Try typing 'loan' to start an application or 'exit' to quit.")