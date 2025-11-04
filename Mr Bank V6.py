# Mr Bank
# AI ChatBot V4

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# === Load and prepare data ===
data_path = r"C:\Users\adamm\Downloads\train.csv"
data = pd.read_csv(data_path)

# Drop missing values for simplicity
data = data.dropna()

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Split data
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === Train model with improved convergence ===
model = LogisticRegression(max_iter=2000, solver='lbfgs')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model trained successfully. Accuracy: {accuracy:.2f}")


# === Chatbot Logic ===
def predict_loan_eligibility(user_input):
    try:
        fields = user_input.split(',')
        if len(fields) != len(X.columns):
            return "Mr Bank: Please provide all required fields separated by commas."

        user_df = pd.DataFrame([fields], columns=X.columns)

        # Convert categorical fields
        for col in user_df.columns:
            if col in label_encoders:
                try:
                    user_df[col] = label_encoders[col].transform(user_df[col])
                except ValueError:
                    return f"Mr Bank: I don't recognise one of your categorical values in '{col}'. Please use known options."
            else:
                user_df[col] = user_df[col].astype(float)

        user_scaled = scaler.transform(user_df)
        prediction = model.predict(user_scaled)[0]

        if prediction == 1:
            return "Mr Bank: Congratulations! Based on your details, your loan is likely to be approved."
        else:
            return "Mr Bank: Unfortunately, your loan eligibility seems low. You may want to review your income or credit history."

    except Exception as e:
        return f"Mr Bank: Hmm, something went wrong processing your input — {str(e)}"


# === Conversation Loop ===
print("\nWelcome to Mr Bank — your AI Loan Assistant (v5.0)")
print("Enter your details in the following order separated by commas:")
print(
    "Gender,Married,Dependents,Education,Employment_Status,Applicant_Income,Coapplicant_Income,Loan_Amount,Loan_Term,Credit_History,Property_Area,Age")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == 'exit':
        print("Mr Bank: Thank you for chatting! Goodbye.")
        break

    response = predict_loan_eligibility(user_input)
    print(response)