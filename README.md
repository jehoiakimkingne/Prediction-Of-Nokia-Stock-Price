# Prediction-Of-Nokia-Stock-Price

- This project is about the application of some Machine Learning algorithms and optimization to predict the end of day stock price of Nokia.
- The aim is to be able to obtain the best prediction of Nokia end of day stock price given the data we have and maybe try to predict the end of day stock price of any given day, given some initial information.
- I will describe the steps I followed in the next lines.

    i- Data Collection and Data Cleaning
    
        - I collected the data necessary for this project using Quandl API.
        
    ii- Data Preparation and Feature Scaling

        - More details are available inside the code
        
    iii- Cross Validation and Training of the model
    
        - I used Principal Component Analysis to selecct the most important features to avoid overfitting
    
    iv- Cross Validation and Training of the model
    
        - As it is a regression problem, I used Random Forest Regressor model from Python sklearn
        
    iv- Tunning Hyperparameters
    
         - I used Random Search for the hyperparameters optimization

- Finally the results are the following:

    - Without hyperparameters optimization, the mean absolute error on the end of day stock price of Nokia is about 0.4$
    - With hyperparameters optimization, the mean absolute error on the end of day stock price of Nokia is about 0.3$
    
- The Principal Component Analysis on all the features show that the most important feature is the Open Day Price. That means for each day, given the Open Day Nokia Stock Price, we can predict the end of day stock price with this model (but trained with more recent data). 
- For futher details, have a look inside the python code.
