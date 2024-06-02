import os
import pickle
import click
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--experiment_name",
    default="nyc-taxi-experiment",
    help="Name of the MLflow experiment"
)
@click.option(
    "--tracking_uri",
    default="sqlite:///mlflow.db",
    help="MLflow tracking URI"
)
def run_train(data_path: str, experiment_name: str, tracking_uri: str):
    # Set the tracking URI
    mlflow.set_tracking_uri(tracking_uri)

    # Set the experiment name
    mlflow.set_experiment(experiment_name)
    
    # Enable MLflow autologging
    mlflow.sklearn.autolog()

    # Start MLflow run
    with mlflow.start_run():
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        print(f"RMSE: {rmse}")


if __name__ == '__main__':
    run_train()
