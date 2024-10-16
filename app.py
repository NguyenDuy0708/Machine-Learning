from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load pre-trained models
mlp_model = joblib.load('MLP.pkl')
lr_model = joblib.load('lr_model.pkl')
ridge_model = joblib.load('ridge_model.pkl')
stacking_model = joblib.load('Stacking.pkl')
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    father_height = float(request.form['father_height'])
    mother_height = float(request.form['mother_height'])
    model_choice = request.form['model']
    if model_choice == 'LinearRegression':
        model = lr_model
    elif model_choice == 'RidgeRegression':
        model = ridge_model
    elif model_choice == 'NeuralNetwork':
        model = mlp_model
    elif model_choice == 'Stacking':
        model = stacking_model
    else:
        return jsonify({'Mô hình được chọn không hợp lệ!'}), 400
    predictions = []
    pred_height = model.predict(np.array([[father_height, mother_height]]).reshape(1, -1))
    predictions.append(pred_height[0])
    return jsonify({'heights': predictions})

if __name__ == '__main__':
    app.run(debug=True)
