import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureAdder(BaseEstimator, TransformerMixin):
    """
    A custom transformer that adds new features based on existing ones in the dataset.

    """

    def __init__(self, add_acc_on_power=True):
        self.add_acc_on_power = add_acc_on_power

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Define indices for feature columns
        # The Indices assume that the dataset will be passed without the `MPG` column
        ACC_INDEX = 4
        HPOWER_INDEX = 2
        CYL_INDEX = 0

        # Calculate new features
        acc_per_cyl = X[:, ACC_INDEX] / X[:, CYL_INDEX]
        acc_per_hp = X[:, ACC_INDEX] / X[:, HPOWER_INDEX]

        return np.c_[X, acc_per_hp, acc_per_cyl]


# Select numerical features
num_attrs = ['Cylinders', 'Displacement', 'Horsepower', 'Weight',
             'Acceleration', 'Model Year']

# Numerical pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('feature_adder', FeatureAdder()),
    ('scaler', StandardScaler()),
])

# Categorical features
cat_attrs = ["Origin"]  # 1: USA, 2: Europe, 3: Asia


# Final Pipeline for preprocessing
preprocessing = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, num_attrs),
        ('cat', OneHotEncoder(), cat_attrs),
    ]
)

# Column Names
cols = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
        'Acceleration', 'Model Year', 'Origin']

num_cols = ['MPG', 'Displacement', 'Horsepower', 'Weight',
            'Acceleration', 'Model Year', 'Cylinders']

cat_cols = ['Origin']

df = pd.read_csv('dataset/auto-mpg.data', names=cols, na_values="?",
                 comment='\t',
                 sep=" ",
                 skipinitialspace=True)

# Train the Model on the Full Dataset
X = df[['Cylinders', 'Displacement', 'Horsepower', 'Weight',
        'Acceleration', 'Model Year', 'Origin']]
y = df[['MPG']]

model = Pipeline([
    ("preprocessing", preprocessing),
    ("random_forest", RandomForestRegressor(max_depth=13,
     max_features=8, min_samples_split=3, random_state=42)),
])

model.fit(X, y)


# export the model into a file
with open("model.pkl", 'wb') as f_out:
    pickle.dump(model, f_out)  # write `model` in .bin file
    f_out.close()  # close the file

with open('model.pkl', 'rb') as f_in:
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


print(model.predict(auto))
