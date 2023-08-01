from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

class ImagePredictor:
    def __init__(self, model_path, test_dir):
        self.model_path = model_path
        self.test_dir = test_dir
        self.model = load_model(self.model_path)
        self.predict_generator = None
        self.predict_results = None
        self.filenames = None
        self.df_cnn = None

    def preprocess_images(self):
        self.predict_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
            directory=self.test_dir,
            target_size=(64, 64),
            batch_size=32,
            class_mode=None,
            shuffle=False
        )
        self.filenames = self.predict_generator.filenames

    def predict(self):
        if self.predict_generator is None:
            raise Exception("Images have not been preprocessed. Please call `preprocess_images` first.")
        self.predict_results = self.model.predict(self.predict_generator)

    def create_dataframe(self):
        if self.predict_results is None:
            raise Exception("No predictions have been made. Please call `predict` first.")
        UserID = [name[2:-4] for name in self.filenames]
        predicted_probabilities = self.predict_results[:, 1]
        self.df_cnn = pd.DataFrame({
            'UserID': UserID,
            'Predicted_Probability': predicted_probabilities
        })