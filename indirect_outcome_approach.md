# I- Description of approach
This approach will consist of running multi stage model. 
* $1^{st}$ stage : Try to find the correct values of NO2_trop.

    $NO2_{trop} = \sum\limits_{i=1}^{n} \beta_i X_i + \epsilon_1$
* $2^{nd}$ stage : Use the NO2_trop as variable now and elevation to find the correct values of GT_NO2

    $NO2_{GT} = \alpha NO2_{trop} + \gamma Elevation + \epsilon_2$


# II - Steps to perform this approach
* Predict the values of $NO2_{GT}$ by using a simple linear regression with $NO2_{trop}$. This will be the baseline.

* Build a model to be able to have the best values of $NO2_{trop}$.

    * Plot the $NO2_{trop}$ to see the evolution:
        
        1- Histplot -> see the distribution.

        2- Line plot with respect to many categories (day, month, year, day of week, season, location)
        
        3- Barplot
        
        4- Pie plot
        
        5 - Density mapbox

        6- Pair plot

        7- Correlation matrix
    
    * Data processing (missing values, outilers, etc)
    
    * Feature engineering
    
    * Scrap the data of elevation
    
    * Build the model to obtain the most accurate values of $NO2_{trop}$

        * Random Forest
        * LGBMRegressor
        * Catboost

* Build a model to predict the values of $NO2_{GT}$
    * Ridge
    * Lasso
    * Random Forest
    * Catboost
    * LGBMRegressor

* Close the approach