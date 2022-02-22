# imports
from os import renames
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import joblib

from memoized_property import memoized_property
import mlflow
from  mlflow.tracking import MlflowClient

from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse, haversine_vectorized
from TaxiFareModel.data import get_data, clean_data

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = "[FR] [Paris] [Joe le Taxi] Model LinearRegression v1"


    def holdout(self, X, y):
        """using train test split of Sklearn to split the data
        train size of 0.3 / random state set to None
        returns X_train, X_test, y_train, y_test"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3)

        return X_train, X_test, y_train, y_test

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        # Distance pipeline
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        # Time pipeline
        time_pipe = Pipeline([
            ('time_encoder', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Preprocessing pipeline
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")

        # Global pipeline
        pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('model', DecisionTreeRegressor())
        ])

        self.pipeline = pipeline

    def run(self, X_train, y_train):
        """set and train the pipeline"""

        # fit the pipeline
        self.pipeline.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        #self.pipeline.evaluate(X_test, y_test)
        y_pred = self.pipeline.predict(X_test)

        return compute_rmse(y_pred, y_test)


    MLFLOW_URI = "https://mlflow.lewagon.co/"

    @memoized_property
    def mlflow_client(self):
        MLFLOW_URI = "https://mlflow.lewagon.co/"
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, 'model.joblib')


if __name__ == "__main__":
    #get data
    df = get_data()
    # clean data
    df_clean = clean_data(df)
    # set X and y
    X = df_clean.drop('fare_amount', axis=1)
    y = df_clean[['fare_amount']].copy()
    # Instantiating my Trainer
    my_trainer = Trainer(X,y)
    # hold out
    X_train, X_test, y_train, y_test = my_trainer.holdout(X,y)
    # train

    my_trainer.set_pipeline()
    my_trainer.run(X_train, y_train)
    # evaluate
    rmse = float(my_trainer.evaluate(X_test, y_test))
    print(f'RMSE final : {rmse}')

    my_trainer.mlflow_log_param('Model', 'Linear')
    my_trainer.mlflow_log_metric('rmse', rmse)
    my_trainer.save_model()
