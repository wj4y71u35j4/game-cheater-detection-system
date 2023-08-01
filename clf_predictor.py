# import pandas as pd
from joblib import load

class ClassifierPredictor:
    def __init__(self, model_path, df_validation):
        self.model_path = model_path
        self.df_validation = df_validation
        self.clf = None

    def load_model(self):
        # Load the model from the file
        self.clf = load(self.model_path)

    def predict(self):
        # Make sure to load the model first
        if self.clf is None:
            self.load_model()

        features = self.df_validation.drop(['ChaNum', 'ChaName', 'UserID'], axis=1)
        predicted_probabilities = self.clf.predict_proba(features)
        self.df_validation['Predicted_Probability'] = predicted_probabilities[:, 1]  # for binary classification
