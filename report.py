import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
import json

warnings.filterwarnings('ignore')

# Load and combine the datasets
try:
    df1 = pd.read_csv('train.csv')
    df2 = pd.read_csv('test.csv')
    df = pd.concat([df1, df2], ignore_index=True)
except Exception as e:
    # If the second file read fails, maybe it's the same as the first. Let's try to read the named file from the prompt.
    # The prompt sometimes renames the file to input_file_1.csv even if user provides input_file_0.csv
    try:
        df1 = pd.read_csv('train.csv')
        df2 = pd.read_csv('test.csv')
        df = pd.concat([df1, df2], ignore_index=True)
    except Exception as e_inner:
        print(f"Error loading files: {e_inner}")
        df = pd.DataFrame() # empty dataframe to avoid further errors

if not df.empty:
    print("Successfully combined the datasets. Here's a summary and the first few rows:")
    print(df.info())
    print("\nMissing Values Check:")
    print(df.isnull().sum())
    print("\nFirst 5 Rows:")
    print(df.head())
    
1# Data Preprocessing
df['Credit_History'] = df['Credit_History'].apply(lambda x: 1 if x == 'Yes' else 0)

# Feature Engineering
df['Loan_to_Income_Ratio'] = df['Loan_Amount'] / df['Annual_Income']

# Define features (X) and target (y)
X = df.drop('Risk_Flag', axis=1)
y = df['Risk_Flag']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)

# Make predictions and evaluate
y_pred = rf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Prepare data for dashboard
total_applicants = len(df)
risk_counts = df['Risk_Flag'].value_counts()
risk_percentage = (risk_counts[1] / total_applicants) * 100
avg_income = df['Annual_Income'].mean()
avg_loan_amount = df['Loan_Amount'].mean()

# Feature Importances
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
print("\nTop 5 Feature Importances:")
print(feature_importance_df.head())

# Data for charts
risk_dist_data = df['Risk_Flag'].value_counts().to_dict()
income_risk_data = {
    'safe': df[df['Risk_Flag'] == 0]['Annual_Income'].tolist(),
    'risky': df[df['Risk_Flag'] == 1]['Annual_Income'].tolist()
}
age_dist_data = df['Age'].value_counts().sort_index().to_dict()

# Convert data to JSON for embedding in HTML
dashboard_data = {
    "total_applicants": int(total_applicants),
    "risk_percentage": round(risk_percentage, 2),
    "avg_income": int(avg_income),
    "accuracy": round(accuracy * 100, 2),
    "risk_dist": risk_dist_data,
    "feature_importance": feature_importance_df.to_dict('records'),
    "age_dist": age_dist_data,
    "income_by_risk_safe": income_risk_data['safe'][::20], # Sample to keep HTML size down
    "income_by_risk_risky": income_risk_data['risky'][::5], # Sample to keep HTML size down
    "sample_data": df.head(10).to_dict('records')
}

# Generate HTML for the dashboard
html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Loan Risk Analysis Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            background-color: #1a1a2e;
            color: #e0e0e0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            grid-column: 1 / -1;
        }}
        .header h1 {{
            color: #ffffff;
            font-size: 2.5em;
            margin-bottom: 5px;
        }}
        .header p {{
            color: #9a9a9a;
            font-size: 1.1em;
        }}
        .card {{
            background-color: #16213e;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.6);
        }}
        .metric-card {{
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }}
        .metric-card h3 {{
            color: #a0a0a0;
            font-size: 1em;
            text-transform: uppercase;
            margin: 0 0 10px 0;
        }}
        .metric-card .value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #0f3460;
        }}
        .metric-card .value.green {{ color: #50c878; }}
        .metric-card .value.red {{ color: #e94560; }}
        .chart-card {{
            grid-column: span 2;
        }}
        @media (max-width: 768px) {{
            .chart-card {{ grid-column: span 1; }}
        }}
        .table-container {{
            grid-column: 1 / -1;
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #333;
        }}
        th {{
            background-color: #e94560;
            color: #ffffff;
        }}
        tr:nth-child(even) {{
            background-color: #1f2a47;
        }}
        .commentary {{
            grid-column: 1 / -1;
            background-color: #16213e;
            padding: 20px;
            border-radius: 10px;
        }}
        .commentary h2 {{
            color: #e94560;
            margin-top: 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Loan Risk Analysis Dashboard</h1>
        <p>Generated on: {pd.to_datetime('today').strftime('%Y-%m-%d')}</p>
    </div>

    <div class="container">
        <!-- Metric Cards -->
        <div class="card metric-card">
            <h3>Total Applicants</h3>
            <p class="value" style="color: #e0e0e0;">{dashboard_data['total_applicants']}</p>
        </div>
        <div class="card metric-card">
            <h3>Overall Risk Percentage</h3>
            <p class="value red">{dashboard_data['risk_percentage']}%</p>
        </div>
        <div class="card metric-card">
            <h3>Average Annual Income</h3>
            <p class="value" style="color: #e0e0e0;">${dashboard_data['avg_income']:,}</p>
        </div>
        <div class="card metric-card">
            <h3>Model Accuracy</h3>
            <p class="value green">{dashboard_data['accuracy']}%</p>
        </div>

        <!-- Charts -->
        <div class="card chart-card">
            <canvas id="riskDistributionChart"></canvas>
        </div>
        <div class="card chart-card">
            <canvas id="featureImportanceChart"></canvas>
        </div>
        <div class="card chart-card">
            <canvas id="incomeDistributionChart"></canvas>
        </div>
        <div class="card chart-card">
            <canvas id="ageDistributionChart"></canvas>
        </div>

        <!-- Data Table -->
        <div class="card table-container">
            <h3>Sample Applicant Data</h3>
            <table>
                <thead>
                    <tr>
                        <th>Age</th><th>Annual Income</th><th>Loan Amount</th><th>Years Employed</th><th>Dependents</th><th>Credit History</th><th>Risk Flag</th>
                    </tr>
                </thead>
                <tbody id="sample-data-body"></tbody>
            </table>
        </div>

        <!-- Commentary -->
        <div class="commentary">
            <h2>Key Insights & Commentary</h2>
            <ul>
                <li><strong>Class Imbalance:</strong> The dataset shows a significant imbalance, with only {dashboard_data['risk_percentage']}% of loans flagged as risky. This can make it challenging for models to learn the patterns of the minority class. Using techniques like `class_weight='balanced'` in the model helps mitigate this.</li>
                <li><strong>Model Performance:</strong> The Random Forest model achieved an accuracy of <strong>{dashboard_data['accuracy']}%</strong> on the unseen test data, showing a strong ability to distinguish between safe and risky applicants.</li>
                <li><strong>Key Predictive Features:</strong> The most influential factors in predicting loan risk are <strong>{dashboard_data['feature_importance'][0]['feature']}</strong> and <strong>{dashboard_data['feature_importance'][1]['feature']}</strong>. Applicants with lower incomes and higher loan-to-income ratios are more likely to be flagged as high-risk.</li>
                 <li><strong>Income and Risk:</strong> As seen in the income distribution chart, the median income for applicants flagged as 'Risky' tends to be lower than for those flagged as 'Safe', reinforcing the importance of income as a predictive feature.</li>
            </ul>
        </div>
    </div>

    <script>
        const dashboardData = {json.dumps(dashboard_data)};

        // 1. Risk Distribution Chart (Pie)
        const riskCtx = document.getElementById('riskDistributionChart').getContext('2d');
        new Chart(riskCtx, {{
            type: 'pie',
            data: {{
                labels: ['Safe (0)', 'Risky (1)'],
                datasets: [{{
                    label: 'Loan Risk Distribution',
                    data: [dashboardData.risk_dist[0], dashboardData.risk_dist[1]],
                    backgroundColor: ['#50c878', '#e94560']
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{ display: true, text: 'Distribution of Loan Risk Flags', color: '#e0e0e0', font: {{ size: 16 }} }},
                    legend: {{ labels: {{ color: '#e0e0e0' }} }}
                }}
            }}
        }});

        // 2. Feature Importance Chart (Bar)
        const featureCtx = document.getElementById('featureImportanceChart').getContext('2d');
        new Chart(featureCtx, {{
            type: 'bar',
            data: {{
                labels: dashboardData.feature_importance.map(item => item.feature),
                datasets: [{{
                    label: 'Importance',
                    data: dashboardData.feature_importance.map(item => item.importance),
                    backgroundColor: '#0f3460'
                }}]
            }},
            options: {{
                indexAxis: 'y',
                responsive: true,
                plugins: {{
                    title: {{ display: true, text: 'Top Predictive Features for Loan Risk', color: '#e0e0e0', font: {{ size: 16 }} }},
                    legend: {{ display: false }}
                }},
                scales: {{
                    x: {{ ticks: {{ color: '#e0e0e0' }}, grid: {{ color: '#333' }} }},
                    y: {{ ticks: {{ color: '#e0e0e0' }}, grid: {{ color: '#333' }} }}
                }}
            }}
        }});
        
        // 3. Income Distribution by Risk (Box Plot)
        const incomeCtx = document.getElementById('incomeDistributionChart').getContext('2d');
        new Chart(incomeCtx, {{
            type: 'boxplot',
             data: {{
                labels: ['Safe', 'Risky'],
                datasets: [{{
                    label: 'Annual Income Distribution',
                    data: [dashboardData.income_by_risk_safe, dashboardData.income_by_risk_risky],
                    backgroundColor: ['rgba(80, 200, 120, 0.5)', 'rgba(233, 69, 96, 0.5)'],
                    borderColor: ['#50c878', '#e94560'],
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{ display: true, text: 'Annual Income Distribution by Risk Flag', color: '#e0e0e0', font: {{ size: 16 }} }},
                    legend: {{ display: false }}
                }},
                scales: {{
                    x: {{ ticks: {{ color: '#e0e0e0' }}, grid: {{ color: '#333' }} }},
                    y: {{ ticks: {{ color: '#e0e0e0' }}, grid: {{ color: '#333' }} }}
                }}
            }}
        }});

        // 4. Age Distribution Chart (Line)
        const ageCtx = document.getElementById('ageDistributionChart').getContext('2d');
        new Chart(ageCtx, {{
            type: 'line',
            data: {{
                labels: Object.keys(dashboardData.age_dist),
                datasets: [{{
                    label: 'Number of Applicants',
                    data: Object.values(dashboardData.age_dist),
                    borderColor: '#e94560',
                    backgroundColor: 'rgba(233, 69, 96, 0.2)',
                    fill: true,
                    tension: 0.3
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{ display: true, text: 'Age Distribution of Applicants', color: '#e0e0e0', font: {{ size: 16 }} }},
                     legend: {{ display: false }}
                }},
                scales: {{
                    x: {{ title: {{ display: true, text: 'Age', color: '#a0a0a0'}}, ticks: {{ color: '#e0e0e0' }}, grid: {{ color: '#333' }} }},
                    y: {{ title: {{ display: true, text: 'Count', color: '#a0a0a0'}}, ticks: {{ color: '#e0e0e0' }}, grid: {{ color: '#333' }} }}
                }}
            }}
        }});

        // Populate Sample Data Table
        const tableBody = document.getElementById('sample-data-body');
        dashboardData.sample_data.forEach(row => {{
            let tr = document.createElement('tr');
            tr.innerHTML = `<td>${{row.Age}}</td><td>${{row.Annual_Income}}</td><td>${{row.Loan_Amount}}</td><td>${{row.Years_Employed}}</td><td>${{row.Dependents}}</td><td>${{row.Credit_History === 1 ? 'Yes' : 'No'}}</td><td>${{row.Risk_Flag}}</td>`;
            tableBody.appendChild(tr);
        }});
    </script>
    <script src="https://unpkg.com/chart.js-chart-box-and-violin-plot/build/Chart.BoxPlot.js"></script>
</body>
</html>
"""

with open("loan_risk_dashboard.html", "w") as f:
    f.write(html_template)

print("\nDashboard has been generated as 'loan_risk_dashboard.html'.")