import pickle
import pandas as pd
from scripts.processing_data import wrangle
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

def make_submissions(test_file_path, model, model_type, time):
    test_data = wrangle(test_file_path)
    submissions = pd.Series(model.predict(test_data), index=test_data.index, name="GT_NO2")
    submissions.to_csv(f"data/submissions/Submission_{model_type}_{time}.csv")
    return None

def save_model(model, model_type, time):
    with open(f"models/model_{model_type}_{time}.pkl", "wb") as f:
        pickle.dump(model, f)
    return None

def check_score(y_true, y_predict):   
    print("RMSE:", root_mean_squared_error(y_true, y_predict))
    print("MAE:", mean_absolute_error(y_true, y_predict))
    print("R²:", r2_score(y_true, y_predict))
    
    return None