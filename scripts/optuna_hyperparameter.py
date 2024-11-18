import optuna
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from category_encoders import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

X_1_train = pd.read_csv("X_1_train.csv", index_col = "ID_Zindi")
y_1_train = pd.read_csv("y_1_train.csv", index_col= "ID_Zindi")
y_1_train = y_1_train.squeeze()

# Objective function for Optuna
def objective(trial):
    # Suggest values for hyperparameters
    strategy = trial.suggest_categorical("strategy", ["mean", "median"])
    max_depth = trial.suggest_int("max_depth", 3, 100)
    max_features = trial.suggest_float("max_features", 0.3,1)
    n_estimators = trial.suggest_int("n_estimators", 200, 3000)

    # Create the pipeline with suggested hyperparameters
    model = make_pipeline(
        OneHotEncoder(use_cat_names=True),
        SimpleImputer(strategy=strategy),
        RandomForestRegressor(
            max_depth=max_depth,
            n_estimators=n_estimators,
	    max_features = max_features,
            random_state=42
        )
    )

    # Perform cross-validation and return the negative MSE
    scores = cross_val_score(model, X_1_train, y_1_train, cv=10, scoring="neg_root_mean_squared_error", n_jobs=30, verbose=3)
    mean_score = np.mean(scores)
    return mean_score  # Optuna maximizes the objective, so no negation needed here

# Set up and run the study
study = optuna.create_study(direction="maximize", study_name="Random Forest")  # We use "maximize" because we want to minimize error
study.optimize(objective, n_trials=100, show_progress_bar=True, n_jobs=-1)

# Best parameters and score
print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
