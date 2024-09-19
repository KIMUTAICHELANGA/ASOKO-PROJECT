import mlflow
import os
import pickle

# Define the path to the models folder
models_folder = 'models'

# Start a new MLflow experiment
mlflow.set_experiment('Model_Experiment')

# Iterate through each .pkl file in the models folder
for model_file in os.listdir(models_folder):
    if model_file.endswith('.pkl'):
        model_path = os.path.join(models_folder, model_file)
        
        # Log the model as an artifact in MLflow
        with mlflow.start_run() as run:
            mlflow.log_artifact(model_path)  # This logs the .pkl file itself
            
            # Optionally, log any additional parameters or metadata
            mlflow.log_param('model_file', model_file)
        
        print(f"Logged model {model_file} to MLflow.")
