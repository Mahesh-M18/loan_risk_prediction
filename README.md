

## Training from Synthetic Data
You can train a model on synthetic loan prediction data using:
```
python src/train_from_synthetic.py
```
This will:
- Generate `train.csv` and `test.csv` if they don't exist
- Train a RandomForest model
- Save the model (`results/rf.joblib`) and preprocessor (`results/preprocessor.joblib`)
- Save a confusion matrix image in `results/confusion_matrix.png`

## Running the App
After training, run:
```
streamlit run app.py
```
Enter applicant details to get a loan risk prediction.

## New Feature: Loan Tenure (months)
- The training pipeline now includes a `Loan_Tenure_Months` feature. If your existing `train.csv`/`test.csv` do not have this column, the script will synthesize a plausible value.
- The Streamlit app exposes a "Loan Tenure (months)" input and uses it during prediction.

## CSV Schema
Required columns for training/inference:
- `Age`, `Annual_Income`, `Loan_Amount`, `Years_Employed`, `Dependents`, `Credit_History` (Yes/No), `Loan_Tenure_Months`, `Risk_Flag` (only for training)

If `Loan_Tenure_Months` is missing, it will be added automatically during training.

---

## Real-World Dataset Workflow
Place your real-world CSVs inside a folder named `real_world_dataset/` at the project root. Filenames containing `train` and `test` will be auto-detected; otherwise the first CSV will be used as training data and a split will be created for testing.

### Train
```
python src/train_real_world.py
```
This will:
- Normalize columns to the model’s schema: `Annual_Income`, `Loan_Amount`, `Loan_Tenure_Months`, `Years_Employed`, `Dependents`, `Credit_History`.
- Build `Risk_Flag` from your dataset if present (e.g., `Default`, `Loan_Status`), otherwise derive it using rules:
  - Money-based risk increases only if Monthly Installment > 0.75 × Monthly Income.
  - Credit history bad (0) adds risk; many dependents and low years employed add risk (same logic as synthetic).
- Train a RandomForest and save artifacts to `results_real/`.

### App
```
streamlit run app_real_world.py
```
- Uses the trained real-world model (`results_real/rf.joblib`).
- Enforces: only show Low Risk if Payment/Income ≤ 0.75×.
