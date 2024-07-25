import pickle

import numpy as np
import pandas as pd
from model import FeatureAdder


with open('model.bin', 'rb') as f_in:
    model = pickle.load(f_in)

# The features must be in this exact order
auto = pd.DataFrame({
    'Cylinders': [4, 6, 8],
    'Displacement': [155.0, 160.0, 165.5],
    'Horsepower': [93.0, 130.0, 98.0],
    'Weight': [2500.0, 3150.0, 2600.0],
    'Acceleration': [15.0, 14.0, 16.0],
    'Model Year': [81, 80, 78],
    'Origin': [3, 2, 1]
})

# MPG: US mpg
# Cylinders: Number of Cyliners
# Displacement: cu. in.
# Horserpower: horsepower
# Weight: lbs
# Acceleration: 0-60 in seconds
# Model Year: 19 _ _
# Origin: 1: USA, 2: Europe, 3: Asia

bmw_m3_coupe_e30_1986 = pd.DataFrame({
    'Cylinders': [4],
    'Displacement': [140.48],
    'Horsepower': [200],
    'Weight': [2568.39],
    'Acceleration': [6.7],
    'Model Year': [86],
    'Origin': [2]
})

print('The Predicted Fuel effeciency of the BMW M3 Coupe E30 1986 is: ',
      round(model.predict(bmw_m3_coupe_e30_1986)[0], 2), ', Actual: 28.34')

porshe_911_coupe_1973 = pd.DataFrame({
    'Cylinders': [6],
    'Displacement': [163.97],
    'Horsepower': [150],
    'Weight': [2425.08],
    'Acceleration': [8.5],
    'Model Year': [73],
    'Origin': [2]
})

print('The Predicted Fuel effeciency of the 1973 Porsche 911 Coupe (G) 2.7 (150 Hp) is: ',
      round(model.predict(porshe_911_coupe_1973)[0], 2), ', Actual: 19.6 - 16.8 US mpg')
