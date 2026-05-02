from flask import Flask, render_template, request
import pickle
import numpy as np

try:
   with open("C:\Program Files\Credit-Card-fraud-detection-all-project\model.pkl", 'rb') as f:
        clf = pickle.load(f)
except Exception as e:
    print("Error loading model:", e)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = [float(x) for x in request.form['features'].split()]
        vect = np.array(features).reshape(1, -1)
        prediction = clf.predict(vect)[0]  # Get the first (and only) prediction
        if prediction == 1:
            prediction_text = "Fraudulent"
        else:
            prediction_text = "Not Fraudulent"
    return render_template('result.html', prediction=prediction_text)

if __name__ == '__main__':
  app.run(debug=True)
