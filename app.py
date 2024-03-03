from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

model_path = 'model.pkl'
model = pickle.load(open(model_path, 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    prediction = model.predict(final_features)
    
    output = 'Malignant' if prediction[0] == 1 else 'Benign'
    
    return render_template('index.html', prediction_text=f'Predicted Cancer Type: {output}')


if __name__ == "__main__":
    app.run(debug=True)
