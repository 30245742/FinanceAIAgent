# Mr Bank
# AI ChatBot V9

# AI Chatbot v8 — Mr Bank Loan Assistant (Stable Build)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


class MrBankChatbot:
    def __init__(self):
        self.model = None
        self.encoders = {}
        self.columns = [
            'Gender', 'Married', 'Dependents', 'Education', 'Employment_Status',
            'Applicant_Income', 'Coapplicant_Income', 'Loan_Amount',
            'Loan_Term', 'Credit_History', 'Property_Area', 'Age'
        ]
        self.train_model()

    def train_model(self):
        # Dummy dataset for prototype
        data = {
            'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'Married': ['Yes', 'No', 'Yes', 'No', 'Yes'],
            'Dependents': [0, 1, 0, 2, 3],
            'Education': ['Graduate', 'Not Graduate', 'Graduate', 'Graduate', 'Not Graduate'],
            'Employment_Status': ['Employed', 'Self Employed', 'Employed', 'Unemployed', 'Employed'],
            'Applicant_Income': [5000, 3000, 4000, 2000, 10000],
            'Coapplicant_Income': [0, 1500, 1200, 0, 2000],
            'Loan_Amount': [150, 100, 120, 80, 200],
            'Loan_Term': [360, 120, 180, 360, 360],
            'Credit_History': [1, 1, 1, 0, 1],
            'Property_Area': ['Urban', 'Rural', 'Semiurban', 'Urban', 'Rural'],
            'Age': [30, 25, 28, 40, 35],
            'Loan_Status': ['Y', 'N', 'Y', 'N', 'Y']
        }

        df = pd.DataFrame(data)
        for col in df.select_dtypes(include=['object']).columns:
            if col != 'Loan_Status':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.encoders[col] = le

        X = df[self.columns]
        y = df['Loan_Status'].apply(lambda x: 1 if x == 'Y' else 0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Mr Bank trained successfully. Model accuracy: {acc:.2f}")

    def normalise_input(self, col, value):
        """Handle case-insensitivity and map to valid label encoder value."""
        if col in self.encoders:
            le = self.encoders[col]
            classes = [c.lower() for c in le.classes_]
            if value.lower() in classes:
                return le.transform([le.classes_[classes.index(value.lower())]])[0]
            else:
                return None
        else:
            try:
                return float(value)
            except ValueError:
                return None

    def get_valid_input(self, col, question):
        """Keep re-asking until valid input."""
        while True:
            value = input(f"Mr Bank: {question}\nYou: ")
            val = self.normalise_input(col, value)
            if val is not None:
                return val
            print(f"Mr Bank: Sorry, I didn’t understand '{value}'. Could you try again?")

    def start_chat(self):
        print("\nMr Bank: Hello! I’m Mr Bank, your AI loan assistant.")
        while True:
            user_input = input("You: ").strip().lower()

            if "loan" in user_input or "apply" in user_input or "approval" in user_input:
                print("Mr Bank: Great! Let’s check your eligibility. Please answer a few questions.\n")

                inputs = {}
                questions = {
                    'Gender': "What is your gender? (Male/Female)",
                    'Married': "Are you married? (Yes/No)",
                    'Dependents': "How many dependents do you have? (0/1/2/3+)",
                    'Education': "What is your education level? (Graduate/Not Graduate)",
                    'Employment_Status': "What is your employment status? (Employed/Self Employed/Unemployed)",
                    'Applicant_Income': "What is your monthly income? (e.g. 4000)",
                    'Coapplicant_Income': "What is your coapplicant's income? (0 if none)",
                    'Loan_Amount': "How much loan amount are you requesting? (in thousands)",
                    'Loan_Term': "What is your loan term? (in months, e.g. 360)",
                    'Credit_History': "Do you have a good credit history? (1 for Yes, 0 for No)",
                    'Property_Area': "Where is the property located? (Urban/Rural/Semiurban)",
                    'Age': "What is your age?"
                }

                for col, q in questions.items():
                    inputs[col] = self.get_valid_input(col, q)

                df_input = pd.DataFrame([inputs])
                prediction = self.model.predict(df_input)[0]
                result = "Approved ✅" if prediction == 1 else "Not Approved ❌"
                print(f"\nMr Bank: Your loan application result is: {result}\n")

            elif user_input in ["exit", "quit", "bye"]:
                print("Mr Bank: Goodbye! Have a nice day.")
                break
            else:
                print("Mr Bank: I can help you check loan eligibility. Just say 'loan approval' to begin.")


if __name__ == "__main__":
    chatbot = MrBankChatbot()
    chatbot.start_chat()
