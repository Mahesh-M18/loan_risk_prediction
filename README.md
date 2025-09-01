# 🏦 Loan Risk Prediction System

## 📖 Project Overview  
**Loan Risk Prediction** is a machine learning project that predicts whether a loan should be approved based on applicant details such as income, education, employment, credit history, and more.  

The project uses **Logistic Regression (best performing model ~79.8% accuracy)** to classify loan applications as **Approved (Y)** or **Not Approved (N)**. A **Streamlit web app** provides an interactive interface for users to enter loan applicant details and get predictions instantly.

---

## 🚀 Key Features

### 🔮 Loan Eligibility Prediction
- Predicts **Loan Approval Status** (`Approved` or `Not Approved`)  
- Displays prediction confidence (probability of approval)  

### 📊 Machine Learning Models
- Logistic Regression (best performing model ~79.8% accuracy)  
- Random Forest, Gradient Boosting, and XGBoost (compared for performance)  

### 🖥️ Streamlit Web Application
- User-friendly UI to input applicant details  
- Real-time loan approval prediction  
- Interactive results with confidence scores  

### 📑 Dataset Preprocessing
- Handling missing values (mode/median imputation)  
- Encoding categorical features (`Gender`, `Education`, `Self Employed`, etc.)  
- Scaling and data transformation  

---

## 🛠️ Tech Stack

| Layer        | Technologies |
|--------------|--------------|
| Programming  | Python |
| ML Libraries | scikit-learn, XGBoost, joblib |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Environment | Jupyter Notebook |

---

## 🗂️ Project Structure
loan_risk_prediction/
├── app.py # Streamlit app
├── loan_model.pkl # Trained Logistic Regression model
├── loan_prediction.ipynb # Jupyter Notebook (EDA + training)
├── requirements.txt # Dependencies
├── results/ # Accuracy, confusion matrix, reports
├── README.md # Project Documentation
└── .gitignore # Git ignored files


---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Mahesh-M18/loan_risk_prediction.git
cd loan_risk_prediction

2. Create a virtual environment

python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows

3. Install dependencies
pip install -r requirements.txt

4. Train and save model (if not already available)
jupyter notebook loan_prediction.ipynb

5. Run the Streamlit app
streamlit run app.py

App will run at: http://127.0.0.1:8501/

📊 Results

Logistic Regression Accuracy: ~79.8% ✅ (Best Model)

Random Forest: ~78.3%

Gradient Boosting: ~77.8%

XGBoost: ~78.3%




🎯 Future Enhancements

Hyperparameter tuning for better accuracy

Deployment on cloud platforms (Heroku, Render, Streamlit Cloud)

Advanced feature engineering

Model explainability (SHAP/feature importance)

Support for batch predictions via CSV upload




🤝 Contributing

We welcome contributions! To contribute:

Fork the repository

Create a new branch (feature/your-feature-name)

Make your changes and commit

Push to your fork and create a Pull Request






📬 Contact & Credits

Developed by

Mahesh M – GitHub

Collaborators welcome!