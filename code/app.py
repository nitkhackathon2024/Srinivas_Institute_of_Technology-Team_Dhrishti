from flask import Flask, request, render_template
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from qiskit_machine_learning.exceptions import QiskitMachineLearningError

vqc_model = joblib.load('vqc_model.pkl')
scaler = joblib.load('scaler.pkl')
app = Flask(__name__)

def preprocess_input(data):
    data = np.array(data).reshape(1, -1)
    num_features = data.shape[1]
    if num_features < 32:
        zeros = np.zeros((1, 32 - num_features))
        data = np.hstack((data, zeros))
    data_scaled = scaler.transform(data)
    if np.isnan(data_scaled).any() or np.isinf(data_scaled).any():
        raise ValueError("Input data contains NaN or infinite values.")
    return data_scaled

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.get('features', '')
    if not data:
        return render_template('index.html', error="No input features provided.")
    try:
        data = [float(x.strip()) for x in data.split(',') if x.strip()]
        preprocessed_data = preprocess_input(data)
    except ValueError as e:
        return render_template('index.html', error=str(e))
    try:
        prediction = vqc_model.predict(preprocessed_data)
        result = "Fraud" if prediction == 1 else "Normal"
    except QiskitMachineLearningError as e:
        return render_template('index.html', error=str(e))
    return render_template('index.html', prediction=result)
if __name__ == '__main__':
    app.run(debug=True)
