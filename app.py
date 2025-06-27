from flask import Flask, render_template, request, session, redirect, url_for, send_file
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from fpdf import FPDF
from datetime import datetime
import secrets
import os

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = secrets.token_hex(16)

# Load dataset
df = pd.read_csv('diabetes.csv')
feature_names = df.columns[:-1].tolist()

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('Outcome', axis=1), 
    df['Outcome'], 
    test_size=0.2, 
    random_state=42
)
model = RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42)
model.fit(X_train, y_train)

@app.route('/')
def home():
    if 'patient_id' not in session:
        session['patient_id'] = f"PAT-{secrets.token_hex(4).upper()}"
    if 'theme' not in session:
        session['theme'] = 'light'
    return render_template("index.html",
                         patient_id=session['patient_id'],
                         current_theme=session['theme'])

@app.route('/toggle-theme')
def toggle_theme():
    session['theme'] = 'dark' if session.get('theme', 'light') == 'light' else 'light'
    return redirect(request.referrer or url_for('home'))

@app.route("/data-analysis")
def data_analysis():
    # Generate EDA visuals
    plt.figure(figsize=(8, 6))
    df['Outcome'].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title('Diabetes Distribution')
    pie_chart = plot_to_base64(plt)
    plt.close()

    plt.figure(figsize=(10, 6))
    df.boxplot(column='Glucose', by='Outcome')
    plt.title('Glucose Levels by Diabetes Status')
    plt.suptitle('')
    glucose_boxplot = plot_to_base64(plt)
    plt.close()

    stats = df.describe().to_html(classes='table table-striped')
    
    return render_template("data_analysis.html",
                         pie_chart=pie_chart,
                         glucose_boxplot=glucose_boxplot,
                         stats=stats,
                         current_theme=session.get('theme', 'light'))

@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form.to_dict()
    
    # Prepare input data
    input_data = {field: float(form_data.get(field, 0)) for field in feature_names}
    
    # Store in session
    session['last_prediction'] = {
        **form_data,
        'timestamp': datetime.now().isoformat(),
        'health_data': input_data
    }
    
    # Predict
    try:
        prediction = model.predict(pd.DataFrame([input_data]))[0]
        probability = model.predict_proba(pd.DataFrame([input_data]))[0][1] * 100
    except Exception as e:
        return f"Prediction error: {str(e)}", 400
    
    # Generate visualizations
    charts = {
        'radar': generate_radar_chart({
            'Glucose': input_data['Glucose'],
            'BloodPressure': input_data['BloodPressure'],
            'BMI': input_data['BMI'],
            'Age': input_data['Age']
        }),
        'risk': generate_risk_chart({
            'Glucose': input_data['Glucose'],
            'BMI': input_data['BMI'],
            'BloodPressure': input_data['BloodPressure'],
            'Age': input_data['Age']
        })
    }
    
    return render_template("result.html",
                         prediction=prediction,
                         probability=f"{probability:.1f}%",
                         charts=charts,
                         factors=analyze_risk_factors(input_data),
                         patient_name=form_data.get('FullName', ''),
                         bmi_explanation=get_bmi_explanation(input_data['BMI']),
                         current_theme=session.get('theme', 'light'))

@app.route("/report")
def generate_report():
    if 'last_prediction' not in session:
        return redirect(url_for('home'))
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Header
    pdf.cell(200, 10, txt="DIABETES RISK REPORT", ln=1, align='C')
    pdf.cell(200, 10, txt=f"Patient: {session['last_prediction'].get('FullName', 'N/A')}", ln=1)
    pdf.cell(200, 10, txt=f"ID: {session['patient_id']}", ln=1)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=1)
    
    # Health Data
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Health Metrics:", ln=1)
    pdf.set_font("Arial", size=10)
    
    health_data = session['last_prediction']['health_data']
    for field in ['Glucose', 'BloodPressure', 'BMI', 'Age']:
        pdf.cell(200, 8, txt=f"{field}: {health_data[field]}", ln=1)
    
    # Risk Factors
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Risk Factors:", ln=1)
    pdf.set_font("Arial", size=10)
    
    factors = analyze_risk_factors(health_data)
    if factors:
        for factor in factors:
            pdf.cell(200, 8, txt=f"- {factor}", ln=1)
    else:
        pdf.cell(200, 8, txt="No significant risk factors detected", ln=1)
    
    report = BytesIO()
    pdf.output(report)
    report.seek(0)
    
    return send_file(
        report,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f"diabetes_report_{session['patient_id']}.pdf"
    )

def plot_to_base64(plt):
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

def get_bmi_explanation(bmi):
    categories = [
        (16, "Severe Thinness"),
        (17, "Moderate Thinness"),
        (18.5, "Mild Thinness"),
        (25, "Normal range"),
        (30, "Overweight"),
        (35, "Obese Class I"),
        (40, "Obese Class II"),
        (float('inf'), "Obese Class III")
    ]
    for limit, category in categories:
        if bmi < limit:
            return f"BMI {bmi:.1f} ({category}) - Weight(kg)/Height(m)²"
    return "BMI classification not available"

def analyze_risk_factors(data):
    factors = []
    if data['Glucose'] > 140:
        factors.append(f"High glucose ({data['Glucose']} mg/dL > 140)")
    if data['BMI'] > 30:
        factors.append(f"Obese BMI ({data['BMI']:.1f} > 30)")
    if data['Age'] > 45:
        factors.append(f"Age over 45 ({data['Age']} years)")
    if data['BloodPressure'] > 90:
        factors.append(f"Elevated BP ({data['BloodPressure']} mmHg > 90)")
    return factors

def generate_radar_chart(data):
    categories = list(data.keys())
    values = list(data.values())
    
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    values += values[:1]
    
    ax.plot(angles, values, color='#4a6fa5', linewidth=2)
    ax.fill(angles, values, color='#4a6fa5', alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title("Health Metrics Radar", pad=20)
    
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

def generate_risk_chart(data):
    factors = {
        'Glucose': (data['Glucose'], 70, 140),
        'BMI': (data['BMI'], 18.5, 25),
        'Blood Pressure': (data['BloodPressure'], 60, 90),
        'Age': (data['Age'], 20, 45)
    }
    
    fig, ax = plt.subplots(figsize=(8,4))
    for name, (value, low, high) in factors.items():
        color = '#e74c3c' if (value < low or value > high) else '#2ecc71'
        ax.barh(name, value, color=color)
        ax.axvline(low, color='#3498db', linestyle='--')
        ax.axvline(high, color='#3498db', linestyle='--')
    
    ax.set_title("Risk Factor Analysis")
    ax.set_xlabel("Value")
    ax.grid(axis='x', alpha=0.3)
    
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

if __name__ == "__main__":
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)