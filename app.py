from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the saved model - fixed by opening in binary mode
try:
    with open(r'D:\GitHub\Heart-Failure-Prediction-using-Machine-Learning\finalized_model.sav', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    print("Error: Model file not found. Please check the file path.")
    exit(1)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    exit(1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form - only include the 7 features used in training
        features = [
            float(request.form['age']),
            float(request.form['cholesterol']),
            float(request.form['max_hr']),
            float(request.form['oldpeak'])
        ]
        
        # Convert categorical features
        # Only include the categorical features used in training
        sex = 1 if request.form['sex'] == 'M' else 0
        exercise_angina = 1 if request.form['exercise_angina'] == 'Y' else 0
        st_slope = {
            'Up': 0, 'Flat': 1, 'Down': 2
        }[request.form['st_slope']]
        
        # Combine all features in the correct order
        features = [features[0], sex, features[1], features[2], exercise_angina, features[3], st_slope]
        
        # Make prediction
        prediction = model.predict([features])
        probability = model.predict_proba([features])[0][1]
        
        result = {
            'prediction': 'High Risk' if prediction[0] == 1 else 'Low Risk',
            'probability': f"{probability*100:.2f}%"
        }
        
        return render_template('result.html', result=result)
    
    except Exception as e:
        return render_template('result.html', result={'prediction': 'Error', 'probability': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 