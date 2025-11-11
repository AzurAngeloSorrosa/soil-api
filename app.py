import os
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
rf = joblib.load('random_forest_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([data['features']])
    pred = rf.predict(features)
    return jsonify({'recommendation': int(pred[0])})

# Use port from environment variable for cloud
port = int(os.environ.get("PORT", 5000))
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
