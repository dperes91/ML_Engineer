import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split, validation_curve
import pycaret.classification as pc
import os
from sklearn.metrics import log_loss, f1_score

# Para usar o sqlite como repositorio
mlflow.set_tracking_uri("sqlite:///mlruns.db")

experiment_name = 'Arremessos_Kobe'
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    experiment = mlflow.get_experiment(experiment_id)
experiment_id = experiment.experiment_id

cols = ['lat','lon','minutes_remaining', 'period','playoffs','shot_distance']

with mlflow.start_run(experiment_id=experiment_id, run_name = 'PipelineAplicacao'):

    model_uri = f"models:/arremessos_kobe@staging"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    data_prod = pd.read_parquet('raw/dataset_kobe_prod.parquet')

    # Remover linhas com valores NaN
    data_prod = data_prod.dropna()
    data_prod = data_prod.reset_index(drop=True)
    data_prod = data_prod.drop_duplicates()


    Y = loaded_model.predict_proba(data_prod[cols])[:,1]
    data_prod['predict_score'] = Y
    true_labels = data_prod['shot_made_flag']


    if len(true_labels) == 0:
        print("Todos os valores verdadeiros são NaN, não é possível calcular as métricas.")
    else:
        data_prod.to_parquet('processed/prediction_prod.parquet')
        mlflow.log_artifact('processed/prediction_prod.parquet')

    print(data_prod)

    mlflow.log_metrics({
        'log_loss_app': log_loss(true_labels, Y),
        'f1_app': f1_score(true_labels, Y > 0.5)
    })