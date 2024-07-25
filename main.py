import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
from model import FeatureAdder

# Initialize Flask and set the template folder to "template"
app = Flask(__name__, template_folder='template')

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Define the home route


@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route


@app.route('/', methods=['POST'])
def predict():
    # Extract form values
    form_values = request.form.to_dict()

    # Convert form values to appropriate types
    features = {
        'Cylinders': [int(form_values['Cylinders'])],
        'Displacement': [float(form_values['Displacement'])],
        'Horsepower': [int(form_values['Horsepower'])],
        'Weight': [float(form_values['Weight'])],
        'Acceleration': [float(form_values['Acceleration'])],
        'Model Year': [int(form_values['Model_Year'])],
        'Origin': [int(form_values['Origin'])]
    }

    # Create a DataFrame
    input_data = pd.DataFrame(features)

    # Predict the mileage
    prediction = model.predict(input_data)

    # Round the output to 2 decimal places
    output = round(prediction[0], 2)

    # Render the result
    if output < 0:
        return render_template('index.html', prediction_text="Predicted Mileage is negative, values entered are not reasonable")
    else:
        return render_template('index.html', prediction_text='Predicted Car Mileage is: {}'.format(output))


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
