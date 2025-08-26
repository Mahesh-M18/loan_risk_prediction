import pandas as pd
import numpy as np
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Fixed random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Check if CSVs exist, else generate synthetic data
if not (os.path.exists("train.csv") and os.path.exists("test.csv")):
    print("Generating synthetic dataset...")
    n_samples = 10000
    ages = np.random.randint(18, 65, size=n_samples)
    annual_income = np.random.randint(20000, 150000, size=n_samples)
    loan_amount = np.random.randint(1000, 50000, size=n_samples)
    years_employed = np.random.randint(0, 40, size=n_samples)
    dependents = np.random.randint(0, 5, size=n_samples)
    credit_history = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.8, 0.2])
    # Loan tenure in months (typical personal loan: 12-84 months)
    loan_tenure_months = np.random.randint(12, 85, size=n_samples)

    risk_flag = []

    for i in range(n_samples):
        risk_score = 0

        monthly_installment = loan_amount[i] / max(loan_tenure_months[i], 1)  # EMI
        monthly_income = annual_income[i] / 12
        
        if ages>=70:
            risk_score+=2
            
        if ages>=50:
            risk_score+=1
            
        if monthly_installment > 0.75 * monthly_income:
            risk_score += 2

        if credit_history[i] == 'No':
            risk_score += 2

        if years_employed[i] < 1:
            risk_score += 2
            
        elif years_employed[i] < 3:
            risk_score += 1

        if dependents[i] >= 3:
            risk_score += 1

        if loan_tenure_months[i] >= 72:
            risk_score += 1

        if np.random.rand() < 0.03:
            risk_score += np.random.choice([-1, 1])

        risk_flag.append(1 if risk_score >= 4 else 0)



    df = pd.DataFrame({
        'Age': ages,
        'Annual_Income': annual_income,
        'Loan_Amount': loan_amount,
        'Years_Employed': years_employed,
        'Dependents': dependents,
        'Credit_History': credit_history,
        'Loan_Tenure_Months': loan_tenure_months,
        'Risk_Flag': risk_flag
    })

    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=df['Risk_Flag'])

    train_df.to_csv("train.csv", index=False)
    test_df.to_csv("test.csv", index=False)
    print("Synthetic dataset saved to train.csv and test.csv")
else:
    print("Using existing train.csv and test.csv")
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    # If tenure column is missing, synthesize a plausible value (12-84 months)
    def add_tenure_if_missing(df: pd.DataFrame) -> pd.DataFrame:
        if 'Loan_Tenure_Months' not in df.columns:
            monthly_income = df['Annual_Income'] / 12
            # Base months proportional to payment burden, clipped to [12, 84]
            base_months = (df['Loan_Amount'] / monthly_income).clip(lower=12, upper=84)
            noise = np.random.randint(-6, 7, size=len(df))
            tenure = (base_months + noise).clip(lower=12, upper=84).astype(int)
            df = df.copy()
            df['Loan_Tenure_Months'] = tenure
        return df

    train_df = add_tenure_if_missing(train_df)
    test_df = add_tenure_if_missing(test_df)

# Encode categorical column
le = LabelEncoder()
train_df['Credit_History'] = le.fit_transform(train_df['Credit_History'])
test_df['Credit_History'] = le.transform(test_df['Credit_History'])

# Split into X, y
X_train = train_df.drop('Risk_Flag', axis=1)
y_train = train_df['Risk_Flag']
X_test = test_df.drop('Risk_Flag', axis=1)
y_test = test_df['Risk_Flag']

# Train RandomForest
print("Training RandomForest model...")
model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
model.fit(X_train, y_train)

# Predictions & evaluation
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Risk','High Risk'], yticklabels=['Low Risk','High Risk'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
os.makedirs("results", exist_ok=True)
cm_path = os.path.join("results", "confusion_matrix.png")
plt.savefig(cm_path)
print(f"Confusion matrix saved to {cm_path}")

# Save model & preprocessor
joblib.dump(model, "results/rf.joblib")
joblib.dump(le, "results/preprocessor.joblib")
print("Model and preprocessor saved to results/")
