import numpy as np
import pandas as pd
import pickle



def wrangle(data_frame_path, is_test_data=True):
    data_frame = pd.read_csv(data_frame_path)
    # Drop useless columns
    data_frame = data_frame.drop(["LST", "ID"], axis = 1)
    # Delete any duplicated observation by grouping them and take the last one
    data_frame = data_frame.groupby("ID_Zindi").last()
    # Convert the date column to a date format
    data_frame["Date"] = pd.to_datetime(data_frame["Date"])
    # Features engineering related to time variables
    data_frame["month"] = data_frame["Date"].dt.month_name()
    data_frame["year"] = data_frame["Date"].dt.year
    data_frame["day"] = data_frame["Date"].dt.day
    data_frame["quarter"] = data_frame["Date"].dt.quarter
    data_frame["is_quarter_start"] = data_frame["Date"].dt.is_quarter_start
    data_frame["is_quarter_end"] = data_frame["Date"].dt.is_quarter_end
    data_frame["is_month_start"] = data_frame["Date"].dt.is_month_start
    data_frame["is_month_end"] = data_frame["Date"].dt.is_month_end
    data_frame["is_year_start"] = data_frame["Date"].dt.is_year_start
    data_frame["is_year_end"] = data_frame["Date"].dt.is_year_end
    data_frame["day_of_week"] = data_frame["Date"].dt.day_name()
    data_frame["day_of_year"] = data_frame["Date"].dt.day_of_year
    data_frame["days_in_month"] = data_frame["Date"].dt.days_in_month
    data_frame["day_trend"] = data_frame["Date"].apply(lambda x:
                                                "Decrease" if 1 <= int(x.strftime("%d")) < 8
                                                else "Increase"
                                                )
    data_frame["month_trend"] = data_frame["Date"].apply(lambda x:
                                                "Decrease" if 1 <= int(x.strftime("%m")) <= 8
                                                else "Increase" if 9 <= int(x.strftime("%m")) <= 12
                                                else "Flat" 
                                                )
    data_frame["season"] = data_frame["Date"].apply(lambda x: 
                                        "fall" if ("09-21" <= x.strftime("%m-%d") <= "12-20")
                                        else "spring" if ("03-21" <= x.strftime("%m-%d") <= "06-20")
                                        else "summer" if ("06-21" <= x.strftime("%m-%d") <= "09-20")
                                        else "winter"
                                        )
    data_frame = data_frame.drop(columns=["Date"])
    
    # Substract the NO2_trop data
        ## Deleting the useless variable GT_NO2
    if is_test_data==True:
        no2_trop = data_frame
    else:
        no2_trop=data_frame.drop(columns = ["GT_NO2"])
        
        ## Filtering the observations where the NO2_trop is null 
    no2_trop_null = no2_trop[no2_trop["NO2_trop"].isnull()]
    no2_trop_null = no2_trop_null.drop(columns=["NO2_trop", "LAT", "LON", "day_trend"])
        ## Filtering the observations where the NO2_trop is not null
    no2_trop_notnull = no2_trop[no2_trop["NO2_trop"].notnull()]["NO2_trop"]
   
    # Find the missing values for NO2_trop
        ## Load the model trop_model_rfr back
    with open('models/trop_model_rfr.pkl', 'rb') as file:
        trop_model_rfr = pickle.load(file)
        ## Predict the values for NO2_trop
    no2_trop_predict = pd.Series(trop_model_rfr.predict(no2_trop_null), index=no2_trop_null.index, name="NO2_trop")
        ## Add the column of NO2_strat found in the to X matrix previously defined
    #no2_trop_null = pd.concat([no2_trop_null, no2_trop_predict], axis=1)
    # Add the predicted values of NO2_trop in the original data set
    no2_trop = pd.concat([no2_trop_notnull, no2_trop_predict])
    
    # Build the final data
    data_frame = data_frame.drop("NO2_trop", axis=1)
    data_frame.insert(8, "NO2_trop", no2_trop)
    
    data_frame = data_frame.drop(columns=["LAT", "LON", "Precipitation", "AAI", "CloudFraction",	"NO2_strat", "NO2_total"])
    
    return data_frame