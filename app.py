from flask import Flask, render_template, request
import numpy as np
import pickle

# Initialize the flask app
app = Flask(__name__)

# Load the model
filename = 'knn_model.pkl'
model = pickle.load(open(filename, 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        Clump_thickness = int(request.form['clump_thickness'])
        Uniformity_of_cell_size = int(request.form['uniformity_of_cell_size'])
        Uniformity_of_cell_shape = int(request.form['uniformity_of_cell_shape'])
        Marginal_adhesion = int(request.form['marginal_adhesion'])
        Bare_nuclei = int(request.form['bare_nuclei'])
        Bland_chromatin = int(request.form['bland_chromatin'])
        Mitoses = int(request.form['mitoses'])

        input_data = np.array([[Clump_thickness, Uniformity_of_cell_size, Uniformity_of_cell_shape, Marginal_adhesion, Bare_nuclei, Bland_chromatin, Mitoses]])
        print("Input data for prediction:", input_data)

        pred = model.predict(input_data)
        print("Prediction result:", pred)

        # Mapping the prediction to the appropriate label
        if pred[0] == 2:
            result = "Benign"
        elif pred[0] == 4:
            result = "Malignant"
        else:
            result = "Unknown"  # Handle unexpected values

        return render_template('index.html', predict=result)
    except Exception as e:
        print("Error occurred:", e)
        return render_template('index.html', predict="An error occurred. Please try again.")

if __name__ == '__main__':
    app.run(debug=True)
