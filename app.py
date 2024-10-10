from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model_filename = 'random_forest_wine_quality.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    features = [float(x) for x in request.form.values()]
    features_array = np.array(features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features_array)
    
    # Determine result
    if prediction[0] == 1:
        result = "This wine is predicted to be of good quality."
    else:
        result = "This wine is predicted to be of poor quality."

    return render_template('result.html', prediction_text=result)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
